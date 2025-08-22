# MFEM AD

This repository provides a general AD interface for the MFEM library

    
## MFEM binding

To configure the project, define your `user.cmake` file in the root directory of the project.
Then put

```
set(MFEM_RELEASE_DIR "/path/to/mfem/release/build")
set(MFEM_DEBUG_DIR "/path/to/mfem/debug/build") 
```

```bash
cmake -DCMAKE_BUILD_TYPE=<Release or Debug> -S . -B <PATH_TO_BUILD>
```

To run examples with parallel support, `MFEM` must be built with `PETSc` support.
