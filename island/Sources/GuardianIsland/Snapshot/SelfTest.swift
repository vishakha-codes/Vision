import AppKit
import Foundation

/// `GuardianIsland --selftest`: drives the real model against a running engine
/// (use the mock engine and SG_ISLAND_HOME / SG_ISLAND_INSECURE_STORE=1), then exits.
@MainActor
enum SelfTest {
    static func run(_ m: IslandModel) {
        var log: [String] = []
        func check(_ ok: Bool, _ what: String) {
            log.append((ok ? "ok   " : "FAIL ") + what)
            print("SELFTEST \(ok ? "ok" : "FAIL"): \(what)")
            if !ok { finish(1) }
        }
        func finish(_ code: Int32) {
            print("SELFTEST done (\(code == 0 ? "pass" : "fail"))")
            fflush(stdout)
            exit(code)
        }
        func after(_ s: Double, _ f: @escaping () -> Void) {
            DispatchQueue.main.asyncAfter(deadline: .now() + s) { f() }
        }
        func type(_ s: String) { for ch in s { m.type(String(ch)) } }

        after(1.5) {
            check(m.connected, "connected to engine socket")
            check(FileManager.default.fileExists(atPath: Paths.guardian.path), "guardian.json written")
            // First run: set a PIN (the setup flow auto-opens).
            if m.lockFlow == nil { m.beginFlow(.setup(confirming: false)) }
            type("12"); m.submit()
            check(m.lockMessage != nil, "rejects a too-short PIN")
            type("2468"); m.submit(); type("2468"); m.submit()
            after(1.2) {
                check(m.lock.hasPassword && m.lock.isUnlocked, "PIN stored (PBKDF2) and unlocked")
                m.request(.setMode(.medium))
                after(0.8) {
                    check(m.mode == .medium, "set_mode with auth accepted by engine")
                    m.request(.pause)
                    after(0.6) {
                        check(m.status == .paused, "pause with auth")
                        m.resume()
                        after(0.6) {
                            check(m.status == .protecting, "resume")
                            m.lock.lockNow()
                            m.request(.setMode(.aggressive))
                            check(m.lockFlow == .unlock(.setMode(.aggressive)), "locked again: asks for PIN")
                            type("0000"); m.submit()
                            after(1.0) {
                                check(m.lockMessage?.contains("left") == true, "wrong PIN rejected")
                                type("2468"); m.submit()
                                after(1.0) {
                                    check(m.lockFlow == nil && m.lock.isUnlocked, "right PIN unlocks")
                                    after(0.8) {
                                        check(m.mode == .aggressive, "pending action ran after unlock")
                                        finish(0)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        after(25) { print("SELFTEST timeout"); finish(2) }
    }
}
