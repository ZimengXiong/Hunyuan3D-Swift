// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Hunyuan3D-Swift",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "Hy3DMLX", targets: ["Hy3DMLX"]),
        .library(name: "HunyuanPaintMLX", targets: ["HunyuanPaintMLX"]),
        .executable(name: "hy3d", targets: ["hy3d"]),
    ],
    dependencies: [
        // Known-good pin: 0.31.4 (rev dc43e62d…) — see Package.resolved. upToNextMinor keeps us on
        // the 0.31.x line (matching the parity-verified Python ports) and off any 0.32 API breakage.
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
    ],
    targets: [
        // MARK: Libraries
        .target(
            name: "Hy3DMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "CXatlas",
            cxxSettings: [.unsafeFlags(["-std=c++14"])]
        ),
        .target(
            name: "HunyuanPaintMLX",
            dependencies: [
                "CXatlas",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),

        // MARK: CLI — shape · paint · generate (chained) · parity-shape · parity-paint
        .executableTarget(
            name: "hy3d",
            dependencies: ["Hy3DMLX", "HunyuanPaintMLX"]
        ),

        // MARK: Parity tests (threshold-gated, fixture-driven; XCTSkip when a fixture is absent)
        .testTarget(
            name: "ShapeParityTests",
            dependencies: ["Hy3DMLX"]
        ),
        .testTarget(
            name: "PaintParityTests",
            dependencies: ["HunyuanPaintMLX"]
        ),
    ]
)
