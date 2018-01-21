# GRTSample Unreal Project

A sample project which porting the [GRT library](https://github.com/nickgillian/grt) into unreal engine module.

Build and test on VS2017 with Unreal Engine 4.18.3.

*Notice: this is some old unfinished work, only DTW(Dynamic Time Warping) classifier will work*

## Why not just use GRT?

You can compile and link GRT with Unreal but there are many reason that they won't work together:

- Many macros get overwritten when link unreal with GRT header
- Unreal Engine use it's own data types to replace the ones in standard C++ library
- Unreal Engine also forbid developer to catch or throw the exception, it turns off unwind semantics by default.
- Unreal Engine has it’s own run-time type identification system and method like "dynamic_cast" will not work.
