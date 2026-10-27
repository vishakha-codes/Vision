import CommonCrypto
import CryptoKit
import Foundation
import Security

/// Generic-password Keychain items under service `ScreenGuardian`.
/// `SG_ISLAND_INSECURE_STORE=1` swaps in plain 0600 files (automated tests only).
enum Keychain {
    static var useFiles: Bool { Paths.env["SG_ISLAND_INSECURE_STORE"] == "1" }

    private static func fileURL(_ account: String) -> URL {
        Paths.supportDir.appendingPathComponent("test-secrets", isDirectory: true).appendingPathComponent(account)
    }

    static func read(_ account: String) -> Data? {
        if useFiles { return try? Data(contentsOf: fileURL(account)) }
        let q: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Paths.keychainService,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var out: CFTypeRef?
        let status = SecItemCopyMatching(q as CFDictionary, &out)
        guard status == errSecSuccess else {
            if status != errSecItemNotFound { NSLog("Screen Guardian island: keychain read \(account) failed (\(status))") }
            return nil
        }
        return out as? Data
    }

    @discardableResult
    static func write(_ account: String, _ data: Data) -> Bool {
        if useFiles {
            let url = fileURL(account)
            try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            return Paths.writePrivate(data, to: url)
        }
        let q: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Paths.keychainService,
            kSecAttrAccount as String: account,
        ]
        let update = SecItemUpdate(q as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if update == errSecSuccess { return true }
        var add = q
        add[kSecValueData as String] = data
        add[kSecAttrLabel as String] = "Screen Guardian (\(account))"
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status = SecItemAdd(add as CFDictionary, nil)
        if status != errSecSuccess { NSLog("Screen Guardian island: keychain write \(account) failed (\(status))") }
        return status == errSecSuccess
    }
}

enum Crypto {
    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func randomBytes(_ n: Int) -> Data {
        var d = Data(count: n)
        let ok = d.withUnsafeMutableBytes { SecRandomCopyBytes(kSecRandomDefault, n, $0.baseAddress!) }
        precondition(ok == errSecSuccess, "SecRandomCopyBytes failed")
        return d
    }

    static func pbkdf2(_ password: String, salt: Data, iterations: Int, length: Int = 32) -> Data {
        let pw = Array(password.utf8)
        var out = [UInt8](repeating: 0, count: length)
        let status = salt.withUnsafeBytes { saltPtr -> Int32 in
            pw.withUnsafeBufferPointer { pwPtr in
                CCKeyDerivationPBKDF(CCPBKDFAlgorithm(kCCPBKDF2),
                                     pwPtr.baseAddress.map { UnsafeRawPointer($0).assumingMemoryBound(to: CChar.self) },
                                     pw.count,
                                     saltPtr.bindMemory(to: UInt8.self).baseAddress, salt.count,
                                     CCPseudoRandomAlgorithm(kCCPRFHmacAlgSHA256), UInt32(iterations),
                                     &out, length)
            }
        }
        precondition(status == kCCSuccess, "PBKDF2 failed")
        return Data(out)
    }

    static func constantTimeEqual(_ a: Data, _ b: Data) -> Bool {
        guard a.count == b.count else { return false }
        var diff: UInt8 = 0
        for (x, y) in zip(a, b) { diff |= x ^ y }
        return diff == 0
    }
}

/// The guardian token the engine checks on guardian actions.
/// Token: 32 random bytes as 64 lowercase hex chars, sent verbatim as `auth`.
/// guardian.json gets sha256(utf8(token)) as hex in `token_sha256`.
enum GuardianToken {
    static let account = "guardian-token"

    static func ensure() -> String? {
        var token: String?
        if let d = Keychain.read(account), let s = String(data: d, encoding: .utf8), s.count == 64 {
            token = s
        } else {
            let fresh = Crypto.randomBytes(32).map { String(format: "%02x", $0) }.joined()
            if Keychain.write(account, Data(fresh.utf8)) { token = fresh }
        }
        if let t = token { publishHash(of: t) }
        return token
    }

    static func publishHash(of token: String) {
        let hash = Crypto.sha256Hex(Data(token.utf8))
        var obj: [String: Any] = [:]
        if let data = try? Data(contentsOf: Paths.guardian),
           let existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            obj = existing
        }
        if obj["token_sha256"] as? String == hash { return }
        obj["token_sha256"] = hash
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]) {
            Paths.writePrivate(data, to: Paths.guardian)
        }
    }
}
