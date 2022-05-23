use std::borrow::Cow;

use super::{LinkerFlavor, PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    const FAMILIES: &[Cow<'_, str>] = &[Cow::Borrowed("unix")];
    let opts = TargetOptions {
        abi: "eabihf".into(),
        cpu: "cortex-a9".into(),
        env: "newlib".into(),
        os: "psvita".into(),
        families: FAMILIES.into(),
        vendor: "vitasdk".into(),
        linker_flavor: LinkerFlavor::Gcc,
        linker: Some("arm-vita-eabi-gcc".into()),
        pre_link_args: [(LinkerFlavor::Gcc, vec!["-Wl,--emit-relocs".into()])].into(),
        features: "+v7,+vfp3,-d32,+thumb2,+neon,+strict-align".into(),
        executables: true,
        exe_suffix: ".elf".into(),
        relocation_model: RelocModel::DynamicNoPic,
        max_atomic_width: Some(64),
        panic_strategy: PanicStrategy::Abort,
        emit_debug_gdb_scripts: false,
        no_default_libraries: false,
        ..Default::default()
    };
    Target {
        llvm_target: "armv7a-none-eabihf".into(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),
        arch: "arm".into(),
        options: opts,
    }
}
