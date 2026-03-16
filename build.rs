use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Copy SDL2.dll to the output directory so the executable can find it at runtime
    let out_dir = env::var("OUT_DIR").unwrap();
    // OUT_DIR is something like target/release/build/portablegl-xxx/out
    // The executable is in target/release/, so go up 3 levels
    let target_dir = Path::new(&out_dir)
        .ancestors()
        .nth(3)
        .unwrap()
        .to_path_buf();

    let dll_src = Path::new("lib/SDL2.dll");
    if dll_src.exists() {
        let dll_dst = target_dir.join("SDL2.dll");
        if !dll_dst.exists() {
            fs::copy(dll_src, &dll_dst).expect("Failed to copy SDL2.dll");
        }
    }
}
