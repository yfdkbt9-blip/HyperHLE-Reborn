/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
// 64-bit (AArch64) counterpart of lib.cpp. It is a separate file and a
// separate set of exported functions (all prefixed `touchHLE64_`), so the
// 32-bit path is not touched at all.
//
// Differences from lib.cpp:
//  * Guest memory is accessed through a table of function pointers passed in on
//    every run (`touchHLE64_MemVTable`) instead of through exported Rust
//    functions. That keeps this file independent of the main crate's `Mem`.
//  * No page table / fastmem yet: every access is a callback. Slow, but simple
//    and correct; fast paths can be added once the 64-bit memory model exists.
//  * The AArch64 equivalent of TPIDRURO is TPIDRRO_EL0 (iOS keeps the thread's
//    TSD array there); TPIDR_EL0 is also provided.
//  * HLE calls are expected to be `svc #imm16` stubs, like in the 32-bit port.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>

#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/exclusive_monitor.h"

namespace touchHLE::cpu64 {

using VAddr = std::uint64_t;
using Vec128 = std::array<std::uint64_t, 2>;

extern "C" {
// Implemented in Rust (src/cpu64.rs). Both return true on success.
struct touchHLE64_MemVTable {
  void *ctx;
  bool (*read)(void *ctx, std::uint64_t addr, void *out, std::size_t size);
  bool (*write)(void *ctx, std::uint64_t addr, const void *data,
                std::size_t size);
};

// Must match `Context64` in src/cpu64.rs exactly.
struct touchHLE64_Context {
  std::uint64_t x[31];
  std::uint64_t sp;
  std::uint64_t pc;
  std::uint32_t pstate;
  std::uint32_t fpcr;
  std::uint32_t fpsr;
  std::uint32_t pad;
  std::uint64_t v[64]; // 32 vector registers, low half then high half
  std::uint64_t tpidr_el0;
  std::uint64_t tpidrro_el0;
};
}

const auto HaltReasonSvc = Dynarmic::HaltReason::UserDefined1;
const auto HaltReasonUndefinedInstruction = Dynarmic::HaltReason::UserDefined2;
const auto HaltReasonBreakpoint = Dynarmic::HaltReason::UserDefined3;

class Environment final : public Dynarmic::A64::UserCallbacks {
public:
  Dynarmic::A64::Jit *cpu = nullptr;
  touchHLE64_MemVTable mem{};
  std::uint64_t ticks_remaining = 0;
  std::uint32_t halting_svc = 0;
  // PC of the instruction that most recently raised an undefined-instruction
  // or breakpoint exception.
  std::uint64_t last_exception_pc = 0;
  // Address of the most recent failed memory access (for diagnostics).
  std::uint64_t last_fault_addr = 0;

private:
  template <typename T> T Read(VAddr addr) {
    T value{};
    if (!mem.read(mem.ctx, addr, &value, sizeof(T))) {
      last_fault_addr = addr;
      cpu->HaltExecution(Dynarmic::HaltReason::MemoryAbort);
    }
    return value;
  }
  template <typename T> void Write(VAddr addr, T value) {
    if (!mem.write(mem.ctx, addr, &value, sizeof(T))) {
      last_fault_addr = addr;
      cpu->HaltExecution(Dynarmic::HaltReason::MemoryAbort);
    }
  }
  template <typename T> bool WriteExclusive(VAddr addr, T value, T expected) {
    if (Read<T>(addr) != expected) {
      return false;
    }
    Write<T>(addr, value);
    return true;
  }

  std::optional<std::uint32_t> MemoryReadCode(VAddr addr) override {
    std::uint32_t value;
    if (!mem.read(mem.ctx, addr, &value, sizeof(value))) {
      last_fault_addr = addr;
      return std::nullopt;
    }
    return value;
  }

  std::uint8_t MemoryRead8(VAddr a) override { return Read<std::uint8_t>(a); }
  std::uint16_t MemoryRead16(VAddr a) override {
    return Read<std::uint16_t>(a);
  }
  std::uint32_t MemoryRead32(VAddr a) override {
    return Read<std::uint32_t>(a);
  }
  std::uint64_t MemoryRead64(VAddr a) override {
    return Read<std::uint64_t>(a);
  }
  Vec128 MemoryRead128(VAddr a) override { return Read<Vec128>(a); }

  void MemoryWrite8(VAddr a, std::uint8_t v) override { Write(a, v); }
  void MemoryWrite16(VAddr a, std::uint16_t v) override { Write(a, v); }
  void MemoryWrite32(VAddr a, std::uint32_t v) override { Write(a, v); }
  void MemoryWrite64(VAddr a, std::uint64_t v) override { Write(a, v); }
  void MemoryWrite128(VAddr a, Vec128 v) override { Write(a, v); }

  bool MemoryWriteExclusive8(VAddr a, std::uint8_t v,
                             std::uint8_t expected) override {
    return WriteExclusive(a, v, expected);
  }
  bool MemoryWriteExclusive16(VAddr a, std::uint16_t v,
                              std::uint16_t expected) override {
    return WriteExclusive(a, v, expected);
  }
  bool MemoryWriteExclusive32(VAddr a, std::uint32_t v,
                              std::uint32_t expected) override {
    return WriteExclusive(a, v, expected);
  }
  bool MemoryWriteExclusive64(VAddr a, std::uint64_t v,
                              std::uint64_t expected) override {
    return WriteExclusive(a, v, expected);
  }
  bool MemoryWriteExclusive128(VAddr a, Vec128 v, Vec128 expected) override {
    return WriteExclusive(a, v, expected);
  }

  void InterpreterFallback(VAddr pc, size_t num_instructions) override {
    std::fprintf(stderr,
                 "touchHLE64: dynarmic wants an interpreter fallback at %llx "
                 "(%zu instructions); not implemented\n",
                 (unsigned long long)pc, num_instructions);
    std::abort();
  }

  void CallSVC(std::uint32_t svc) override {
    halting_svc = svc;
    cpu->HaltExecution(HaltReasonSvc);
  }

  void ExceptionRaised(VAddr pc, Dynarmic::A64::Exception exception) override {
    using E = Dynarmic::A64::Exception;
    switch (exception) {
    case E::NoExecuteFault:
      // MemoryReadCode returned nullopt
      last_fault_addr = pc;
      cpu->HaltExecution(Dynarmic::HaltReason::MemoryAbort);
      break;
    case E::Breakpoint:
      last_exception_pc = pc;
      cpu->HaltExecution(HaltReasonBreakpoint);
      break;
    case E::Yield:
    case E::WaitForEvent:
    case E::WaitForInterrupt:
    case E::SendEvent:
    case E::SendEventLocal:
      // Hints (YIELD/WFE/WFI/SEV): no-ops for us. Just keep running.
      break;
    default:
      // Undefined / unallocated / unpredictable encodings.
      last_exception_pc = pc;
      cpu->HaltExecution(HaltReasonUndefinedInstruction);
      break;
    }
  }

  void AddTicks(std::uint64_t ticks) override {
    if (ticks > ticks_remaining) {
      ticks_remaining = 0;
      return;
    }
    ticks_remaining -= ticks;
  }
  std::uint64_t GetTicksRemaining() override { return ticks_remaining; }

  // The system counter. iPhones tick this at 24 MHz.
  std::uint64_t GetCNTPCT() override {
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count();
    return (std::uint64_t(ns) / 1000) * 24;
  }
};

class Wrapper {
  Environment env;
  // Compiled guest code reads these through pointers, so they must stay at
  // fixed addresses.
  std::uint64_t tpidr_el0 = 0;
  std::uint64_t tpidrro_el0 = 0;
  std::unique_ptr<Dynarmic::ExclusiveMonitor> mon;
  std::unique_ptr<Dynarmic::A64::Jit> cpu;

public:
  Wrapper() {
    Dynarmic::A64::UserConfig config;
    config.callbacks = &env;
    mon = std::make_unique<Dynarmic::ExclusiveMonitor>(1);
    config.global_monitor = mon.get();
    config.tpidr_el0 = &tpidr_el0;
    config.tpidrro_el0 = &tpidrro_el0;
    // Needed so that HaltExecution() called from a memory callback takes
    // effect right after the access instead of at the end of the block.
    config.check_halt_on_memory_access = true;
    // Needed for AddTicks/GetTicksRemaining to be used.
    config.enable_cycle_counting = true;
    cpu = std::make_unique<Dynarmic::A64::Jit>(config);
    env.cpu = cpu.get();
  }

  // Index 0..=30: X0..X30, 31: SP, 32: PC.
  std::uint64_t get_reg(std::uint32_t i) const {
    if (i < 31)
      return cpu->GetRegister(i);
    return i == 31 ? cpu->GetSP() : cpu->GetPC();
  }
  void set_reg(std::uint32_t i, std::uint64_t v) {
    if (i < 31)
      cpu->SetRegister(i, v);
    else if (i == 31)
      cpu->SetSP(v);
    else
      cpu->SetPC(v);
  }
  void get_vec(std::uint32_t i, std::uint64_t *out) const {
    auto v = cpu->GetVector(i);
    out[0] = v[0];
    out[1] = v[1];
  }
  void set_vec(std::uint32_t i, std::uint64_t lo, std::uint64_t hi) {
    cpu->SetVector(i, Vec128{lo, hi});
  }

  std::uint32_t pstate() const { return cpu->GetPstate(); }
  void set_pstate(std::uint32_t v) { cpu->SetPstate(v); }
  std::uint32_t fpcr() const { return cpu->GetFpcr(); }
  void set_fpcr(std::uint32_t v) { cpu->SetFpcr(v); }
  std::uint32_t fpsr() const { return cpu->GetFpsr(); }
  void set_fpsr(std::uint32_t v) { cpu->SetFpsr(v); }
  void set_tpidr_el0(std::uint64_t v) { tpidr_el0 = v; }
  void set_tpidrro_el0(std::uint64_t v) { tpidrro_el0 = v; }
  std::uint64_t last_exception_pc() const { return env.last_exception_pc; }
  std::uint64_t last_fault_addr() const { return env.last_fault_addr; }

  void invalidate_cache_range(VAddr start, std::uint64_t size) {
    cpu->InvalidateCacheRange(start, size);
  }

  void save_context(touchHLE64_Context *c) const {
    for (std::uint32_t i = 0; i < 31; i++)
      c->x[i] = cpu->GetRegister(i);
    c->sp = cpu->GetSP();
    c->pc = cpu->GetPC();
    c->pstate = cpu->GetPstate();
    c->fpcr = cpu->GetFpcr();
    c->fpsr = cpu->GetFpsr();
    c->pad = 0;
    for (std::uint32_t i = 0; i < 32; i++) {
      auto v = cpu->GetVector(i);
      c->v[2 * i] = v[0];
      c->v[2 * i + 1] = v[1];
    }
    c->tpidr_el0 = tpidr_el0;
    c->tpidrro_el0 = tpidrro_el0;
  }
  void load_context(const touchHLE64_Context *c) {
    for (std::uint32_t i = 0; i < 31; i++)
      cpu->SetRegister(i, c->x[i]);
    cpu->SetSP(c->sp);
    cpu->SetPC(c->pc);
    cpu->SetPstate(c->pstate);
    cpu->SetFpcr(c->fpcr);
    cpu->SetFpsr(c->fpsr);
    for (std::uint32_t i = 0; i < 32; i++)
      cpu->SetVector(i, Vec128{c->v[2 * i], c->v[2 * i + 1]});
    tpidr_el0 = c->tpidr_el0;
    tpidrro_el0 = c->tpidrro_el0;
  }
  void swap_context(touchHLE64_Context *c) {
    touchHLE64_Context tmp;
    save_context(&tmp);
    load_context(c);
    *c = tmp;
  }

  // Return value:
  //   -1: ran out of ticks (or finished a single step)
  //   -2: memory abort (see last_fault_addr)
  //   -3: undefined instruction (see last_exception_pc)
  //   -4: breakpoint (BRK) (see last_exception_pc)
  //  >=0: guest executed `svc #N`; N is returned
  std::int32_t run_or_step(const touchHLE64_MemVTable *mem,
                           std::uint64_t *ticks) {
    env.mem = *mem;
    Dynarmic::HaltReason hr;
    if (ticks) {
      env.ticks_remaining = *ticks;
      hr = cpu->Run();
    } else {
      hr = cpu->Step();
    }
    std::int32_t res;
    if ((!hr && ticks) || (hr == Dynarmic::HaltReason::Step && !ticks)) {
      res = -1;
    } else if (Dynarmic::Has(hr, Dynarmic::HaltReason::MemoryAbort)) {
      res = -2;
    } else if (Dynarmic::Has(hr, HaltReasonUndefinedInstruction)) {
      res = -3;
    } else if (Dynarmic::Has(hr, HaltReasonBreakpoint)) {
      res = -4;
    } else if (Dynarmic::Has(hr, HaltReasonSvc)) {
      res = std::int32_t(env.halting_svc);
    } else {
      std::printf("touchHLE64: unhandled halt reason %u\n", unsigned(hr));
      std::abort();
    }
    env.mem = {};
    if (ticks) {
      *ticks = env.ticks_remaining;
    }
    return res;
  }
};

extern "C" {

Wrapper *touchHLE64_new() { return new Wrapper(); }
void touchHLE64_delete(Wrapper *cpu) { delete cpu; }

std::uint64_t touchHLE64_get_reg(const Wrapper *cpu, std::uint32_t index) {
  return cpu->get_reg(index);
}
void touchHLE64_set_reg(Wrapper *cpu, std::uint32_t index,
                        std::uint64_t value) {
  cpu->set_reg(index, value);
}
void touchHLE64_get_vec(const Wrapper *cpu, std::uint32_t index,
                        std::uint64_t *out) {
  cpu->get_vec(index, out);
}
void touchHLE64_set_vec(Wrapper *cpu, std::uint32_t index, std::uint64_t lo,
                        std::uint64_t hi) {
  cpu->set_vec(index, lo, hi);
}

std::uint32_t touchHLE64_pstate(const Wrapper *cpu) { return cpu->pstate(); }
void touchHLE64_set_pstate(Wrapper *cpu, std::uint32_t v) {
  cpu->set_pstate(v);
}
std::uint32_t touchHLE64_fpcr(const Wrapper *cpu) { return cpu->fpcr(); }
void touchHLE64_set_fpcr(Wrapper *cpu, std::uint32_t v) { cpu->set_fpcr(v); }
std::uint32_t touchHLE64_fpsr(const Wrapper *cpu) { return cpu->fpsr(); }
void touchHLE64_set_fpsr(Wrapper *cpu, std::uint32_t v) { cpu->set_fpsr(v); }
void touchHLE64_set_tpidr_el0(Wrapper *cpu, std::uint64_t v) {
  cpu->set_tpidr_el0(v);
}
void touchHLE64_set_tpidrro_el0(Wrapper *cpu, std::uint64_t v) {
  cpu->set_tpidrro_el0(v);
}
std::uint64_t touchHLE64_last_exception_pc(const Wrapper *cpu) {
  return cpu->last_exception_pc();
}
std::uint64_t touchHLE64_last_fault_addr(const Wrapper *cpu) {
  return cpu->last_fault_addr();
}

void touchHLE64_swap_context(Wrapper *cpu, touchHLE64_Context *context) {
  cpu->swap_context(context);
}
void touchHLE64_invalidate_cache_range(Wrapper *cpu, std::uint64_t start,
                                       std::uint64_t size) {
  cpu->invalidate_cache_range(start, size);
}

std::int32_t touchHLE64_run_or_step(Wrapper *cpu,
                                    const touchHLE64_MemVTable *mem,
                                    std::uint64_t *ticks) {
  return cpu->run_or_step(mem, ticks);
}
}

} // namespace touchHLE::cpu64
