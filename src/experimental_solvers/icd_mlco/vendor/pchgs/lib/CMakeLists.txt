# Integration-only replacement for upstream pchgs/lib/CMakeLists.txt.
# The package versions are unchanged; release archives avoid downloading
# complete repository histories on a fresh benchmark machine.
set(PCHGS_GITHUB_PREFIX "https://github.com" CACHE STRING "GitHub archive mirror prefix")

CPMAddPackage(
        NAME fmt
        URL ${PCHGS_GITHUB_PREFIX}/fmtlib/fmt/archive/refs/tags/8.1.1.tar.gz
        OPTIONS "FMT_INSTALL YES" "FMT_TEST OFF"
)

CPMAddPackage(
        NAME nlohmann_json
        URL ${PCHGS_GITHUB_PREFIX}/nlohmann/json/archive/refs/tags/v3.9.1.tar.gz
        OPTIONS "JSON_BuildTests OFF"
)

CPMAddPackage(
        NAME dynamic_bitset
        URL ${PCHGS_GITHUB_PREFIX}/pinam45/dynamic_bitset/archive/refs/tags/v1.2.1.tar.gz
        OPTIONS
          "DYNAMICBITSET_USE_LIBPOPCNT OFF"
          "DYNAMICBITSET_BUILD_EXAMPLE OFF"
          "DYNAMICBITSET_BUILD_TESTS OFF"
          "DYNAMICBITSET_BUILD_DOCS OFF"
)
