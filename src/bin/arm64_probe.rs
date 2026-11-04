/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
//! Minimal AArch64 entry-point probe for an extracted Mach-O or iOS IPA.
//! This is a bring-up tool, not a complete iOS process launcher: it stops at
//! the first SVC, exception, or unmapped-memory fault.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use touchHLE::arm64::cpu::{Cpu, CpuContext, StopReason};
use touchHLE::arm64::mach_o::MachOImage;
use touchHLE::arm64::memory::GuestMemory;
use touchHLE::arm64::stack::{apply_initial_stack, build_initial_stack};

fn read_bundle_executable(path: &Path) -> Result<(Vec<u8>, String), String> {
    if path.is_dir() {
        let info_path = path.join("Info.plist");
        let info = plist::Value::from_file(&info_path)
            .map_err(|error| format!("cannot read {}: {error}", info_path.display()))?;
        let executable = info.as_dictionary()
            .and_then(|dictionary| dictionary.get("CFBundleExecutable"))
            .and_then(plist::Value::as_string)
            .ok_or("Info.plist has no CFBundleExecutable")?;
        let executable_path = path.join(executable);
        let bytes = std::fs::read(&executable_path)
            .map_err(|error| format!("cannot read {}: {error}", executable_path.display()))?;
        return Ok((bytes, executable.to_owned()));
    }

    if path.extension().and_then(|extension| extension.to_str()) == Some("ipa") {
        let file = File::open(path).map_err(|error| error.to_string())?;
        let mut archive = zip::ZipArchive::new(file).map_err(|error| error.to_string())?;
        let mut info_name = None;
        for index in 0..archive.len() {
            let entry = archive.by_index(index).map_err(|error| error.to_string())?;
            let name = entry.name();
            if name.starts_with("Payload/") && name.ends_with(".app/Info.plist") {
                info_name = Some(name.to_owned());
                break;
            }
        }
        let info_name = info_name.ok_or("IPA contains no Payload/*.app/Info.plist")?;
        let mut info_bytes = Vec::new();
        archive.by_name(&info_name).map_err(|error| error.to_string())?
            .read_to_end(&mut info_bytes).map_err(|error| error.to_string())?;
        let info = plist::Value::from_reader(info_bytes.as_slice())
            .map_err(|error| error.to_string())?;
        let executable = info.as_dictionary()
            .and_then(|dictionary| dictionary.get("CFBundleExecutable"))
            .and_then(plist::Value::as_string)
            .ok_or("Info.plist has no CFBundleExecutable")?;
        let app_prefix = info_name.strip_suffix("Info.plist").ok_or("invalid app path")?;
        let executable_path = format!("{app_prefix}{executable}");
        let mut bytes = Vec::new();
        archive.by_name(&executable_path).map_err(|error| {
            format!("cannot find app executable {executable_path}: {error}")
        })?.read_to_end(&mut bytes).map_err(|error| error.to_string())?;
        return Ok((bytes, executable.to_owned()));
    }

    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("guest").to_owned();
    Ok((bytes, name))
}

fn run(path: PathBuf) -> Result<(), String> {
    let (bytes, executable_name) = read_bundle_executable(&path)?;
    let mut memory = GuestMemory::new();
    let image = MachOImage::load(&bytes, &mut memory, 0x1_0000_0000)
        .map_err(|error| format!("AArch64 Mach-O load failed: {error:?}"))?;

    let stack = build_initial_stack(
        &mut memory,
        0x7000_0000_0000,
        0x20_0000,
        &[&executable_name],
        &["HOME=/var/mobile", "TMPDIR=/tmp", "PATH=/usr/bin:/bin"],
        &["platform=iPhone", "executable_path=/Payload/App.app"],
    ).map_err(|error| format!("cannot construct initial stack: {error:?}"))?;

    let mut context = CpuContext::default();
    context.pc = image.entry_point;
    apply_initial_stack(&mut context, stack);

    println!("AArch64 image loaded: {executable_name}");
    println!("entry={:#018x} text={:#018x} end={:#018x}", image.entry_point, image.text_base, image.end_address);
    println!("segments={} mapped_pages={}", image.segments.len(), memory.mapped_page_count());

    let mut cpu = Cpu::new(&mut memory);
    cpu.set_context(&context);
    let (reason, svc, fault_pc) = cpu.run(10_000_000);
    let final_context = cpu.get_context();
    println!("stop={reason:?} svc={svc:#x} fault_pc={fault_pc:#018x}");
    println!("pc={:#018x} sp={:#018x} x0={:#018x} x1={:#018x}",
        final_context.pc, final_context.sp, final_context.x[0], final_context.x[1]);

    match reason {
        StopReason::Normal => println!("instruction budget exhausted without a stop event"),
        StopReason::SupervisorCall => println!("stopped at SVC; no iOS kernel/framework services are installed in this probe"),
        StopReason::Exception => return Err(format!("guest exception at {fault_pc:#x}")),
        StopReason::MemoryFault => return Err(format!("unmapped guest memory access near {fault_pc:#x}")),
    }
    Ok(())
}

fn main() {
    let mut args = std::env::args_os();
    let _program = args.next();
    let Some(path) = args.next() else {
        eprintln!("usage: arm64-probe <app.ipa | App.app | Mach-O executable>");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("arm64-probe accepts exactly one input path");
        std::process::exit(2);
    }
    if let Err(error) = run(PathBuf::from(path)) {
        eprintln!("arm64-probe: {error}");
        std::process::exit(1);
    }
}
