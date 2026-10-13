// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BibleAICore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "BibleAICore", targets: ["BibleAICore"]),
    ],
    targets: [
        .target(
            name: "BibleAICore",
            linkerSettings: [.linkedLibrary("sqlite3")]
        ),
        .testTarget(
            name: "BibleAICoreTests",
            dependencies: ["BibleAICore"],
            exclude: ["Fixtures"]
        ),
    ]
)
