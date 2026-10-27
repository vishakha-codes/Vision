// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "GuardianIsland",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "GuardianIsland", targets: ["GuardianIsland"]),
    ],
    targets: [
        .executableTarget(
            name: "GuardianIsland",
            path: "Sources/GuardianIsland",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("SwiftUI"),
                .linkedFramework("Security"),
            ]
        ),
    ]
)
