/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include <array>
#include <cstdint>
#include <cstdio>
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/halt_reason.h"

namespace touchHLE::arm64 {
using VAddr = std::uint64_t;
using Vector = Dynarmic::A64::Vector;
struct touchHLE_A64Memory;
extern "C" {
bool touchHLE_a64_read(touchHLE_A64Memory*, VAddr, std::uint8_t*, std::size_t);
bool touchHLE_a64_write(touchHLE_A64Memory*, VAddr, const std::uint8_t*, std::size_t);
}
struct Context {
  std::uint64_t x[31]{};
  std::uint64_t sp = 0;
  std::uint64_t pc = 0;
  Vector v[32]{};
  std::uint32_t fpcr = 0;
  std::uint32_t fpsr = 0;
  std::uint32_t pstate = 0;
};
enum class StopReason : std::uint32_t { Normal=0, Svc=1, Exception=2, Memory=3 };
class Callbacks final : public Dynarmic::A64::UserCallbacks {
public:
  explicit Callbacks(touchHLE_A64Memory* memory) : mem(memory) {}
  Dynarmic::A64::Jit* jit = nullptr;
  touchHLE_A64Memory* mem;
  std::uint64_t ticks = 0;
  std::uint32_t svc = 0;
  StopReason reason = StopReason::Normal;
  std::uint64_t exception_pc = 0;
  std::uint8_t MemoryRead8(VAddr a) override { std::uint8_t v{}; if(!read(a,&v,sizeof(v))) fault(); return v; }
  std::uint16_t MemoryRead16(VAddr a) override { std::uint16_t v{}; if(!read(a,&v,sizeof(v))) fault(); return v; }
  std::uint32_t MemoryRead32(VAddr a) override { std::uint32_t v{}; if(!read(a,&v,sizeof(v))) fault(); return v; }
  std::uint64_t MemoryRead64(VAddr a) override { std::uint64_t v{}; if(!read(a,&v,sizeof(v))) fault(); return v; }
  Vector MemoryRead128(VAddr a) override { Vector v{}; if(!read(a,v.data(),sizeof(v))) fault(); return v; }
  void MemoryWrite8(VAddr a,std::uint8_t v) override { if(!write(a,&v,sizeof(v))) fault(); }
  void MemoryWrite16(VAddr a,std::uint16_t v) override { if(!write(a,&v,sizeof(v))) fault(); }
  void MemoryWrite32(VAddr a,std::uint32_t v) override { if(!write(a,&v,sizeof(v))) fault(); }
  void MemoryWrite64(VAddr a,std::uint64_t v) override { if(!write(a,&v,sizeof(v))) fault(); }
  void MemoryWrite128(VAddr a,Vector v) override { if(!write(a,v.data(),sizeof(v))) fault(); }
  bool MemoryWriteExclusive8(VAddr a,std::uint8_t v,std::uint8_t e) override { return exclusive(a,v,e); }
  bool MemoryWriteExclusive16(VAddr a,std::uint16_t v,std::uint16_t e) override { return exclusive(a,v,e); }
  bool MemoryWriteExclusive32(VAddr a,std::uint32_t v,std::uint32_t e) override { return exclusive(a,v,e); }
  bool MemoryWriteExclusive64(VAddr a,std::uint64_t v,std::uint64_t e) override { return exclusive(a,v,e); }
  bool MemoryWriteExclusive128(VAddr a,Vector v,Vector e) override { return exclusive(a,v,e); }
  void InterpreterFallback(VAddr, size_t) override { reason=StopReason::Exception; exception_pc=jit->GetPC(); jit->HaltExecution(); }
  void CallSVC(std::uint32_t n) override { svc=n; reason=StopReason::Svc; jit->HaltExecution(Dynarmic::HaltReason::UserDefined1); }
  void ExceptionRaised(VAddr pc,Dynarmic::A64::Exception ex) override {
    if(ex==Dynarmic::A64::Exception::WaitForInterrupt || ex==Dynarmic::A64::Exception::WaitForEvent ||
       ex==Dynarmic::A64::Exception::Yield || ex==Dynarmic::A64::Exception::SendEvent ||
       ex==Dynarmic::A64::Exception::SendEventLocal) return;
    exception_pc=pc; reason=StopReason::Exception; jit->HaltExecution(Dynarmic::HaltReason::UserDefined2);
  }
  void AddTicks(std::uint64_t n) override { ticks = n>ticks ? 0 : ticks-n; }
  std::uint64_t GetTicksRemaining() override { return ticks; }
  std::uint64_t GetCNTPCT() override { return 0; }
private:
  bool read(VAddr a,void* p,std::size_t n) { return touchHLE_a64_read(mem,a,static_cast<std::uint8_t*>(p),n); }
  bool write(VAddr a,const void* p,std::size_t n) { return touchHLE_a64_write(mem,a,static_cast<const std::uint8_t*>(p),n); }
  void fault() { reason=StopReason::Memory; jit->HaltExecution(Dynarmic::HaltReason::MemoryAbort); }
  template<typename T> bool exclusive(VAddr a,T v,T expected) { T old{}; if(!read(a,&old,sizeof(old))) { fault(); return false; } if(old!=expected) return false; return write(a,&v,sizeof(v)); }
};
struct Wrapper { Callbacks callbacks; Dynarmic::A64::Jit jit; Wrapper(touchHLE_A64Memory* mem):callbacks(mem),jit([&]{ Dynarmic::A64::UserConfig c{&callbacks}; c.enable_cycle_counting=true; return c;}()) { callbacks.jit=&jit; } };
}
extern "C" {
void* touchHLE_A64_new(void* mem) { return new touchHLE::arm64::Wrapper(static_cast<touchHLE::arm64::touchHLE_A64Memory*>(mem)); }
void touchHLE_A64_delete(void* p) { delete static_cast<touchHLE::arm64::Wrapper*>(p); }
void touchHLE_A64_set_context(void* p,const touchHLE::arm64::Context* c) {
 auto& j=static_cast<touchHLE::arm64::Wrapper*>(p)->jit;
 j.SetRegisters(*reinterpret_cast<const std::array<std::uint64_t,31>*>(c->x));
 j.SetSP(c->sp); j.SetPC(c->pc);
 j.SetVectors(*reinterpret_cast<const std::array<touchHLE::arm64::Vector,32>*>(c->v));
 j.SetFpcr(c->fpcr); j.SetFpsr(c->fpsr); j.SetPstate(c->pstate);
}
void touchHLE_A64_get_context(void* p,touchHLE::arm64::Context* c) {
 auto& j=static_cast<touchHLE::arm64::Wrapper*>(p)->jit;
 auto x=j.GetRegisters(); for(size_t i=0;i<31;i++) c->x[i]=x[i];
 c->sp=j.GetSP(); c->pc=j.GetPC();
 auto v=j.GetVectors(); for(size_t i=0;i<32;i++) c->v[i]=v[i];
 c->fpcr=j.GetFpcr(); c->fpsr=j.GetFpsr(); c->pstate=j.GetPstate();
}
std::uint32_t touchHLE_A64_run(void* p,std::uint64_t ticks,bool step,std::uint32_t* svc,std::uint64_t* exception_pc) {
 auto* w=static_cast<touchHLE::arm64::Wrapper*>(p);
 w->callbacks.ticks=ticks; w->callbacks.reason=touchHLE::arm64::StopReason::Normal;
 auto halt=step?w->jit.Step():w->jit.Run();
 *svc=w->callbacks.svc; *exception_pc=w->callbacks.exception_pc;
 if(w->callbacks.reason!=touchHLE::arm64::StopReason::Normal) return static_cast<std::uint32_t>(w->callbacks.reason);
 return halt==Dynarmic::HaltReason::MemoryAbort?3u:0u;
}
}
