package openclaw

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"go.autonomous.ai/os/system/server/config"
)

func TestRuntimeInfoAtLeastPatch(t *testing.T) {
	cases := []struct {
		r    RuntimeInfo
		want bool
	}{
		{RuntimeInfo{}, true},
		{RuntimeInfo{Year: 2026, Minor: 9, Patch: 3, Detected: true}, true},
		{RuntimeInfo{Year: 2026, Minor: 9, Patch: 34, Detected: true}, true},
		{RuntimeInfo{Year: 2026, Minor: 10, Patch: 1, Detected: true}, true},
		{RuntimeInfo{Year: 2027, Minor: 1, Patch: 1, Detected: true}, true},
		{RuntimeInfo{Year: 2026, Minor: 9, Patch: 2, Detected: true}, false},
		{RuntimeInfo{Year: 2026, Minor: 6, Patch: 10, Detected: true}, false},
		{RuntimeInfo{Year: 2025, Minor: 12, Patch: 40, Detected: true}, false},
	}
	for _, c := range cases {
		if got := c.r.StrictConfig(); got != c.want {
			t.Errorf("%+v StrictConfig() = %v, want %v", c.r, got, c.want)
		}
	}
}

// writeRetiredKeysConfig writes an openclaw.json holding the keys OpenClaw 2026.9.3 retired.
func writeRetiredKeysConfig(t *testing.T) (*OpenclawService, string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "openclaw.json")
	body := `{"agents":{"defaults":{"compaction":{"mode":"safeguard","reserveTokensFloor":5000}}},` +
		`"gateway":{"controlUi":{"allowedOrigins":["http://127.0.0.1","http://localhost"],"allowInsecureAuth":false}}}`
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	return &OpenclawService{config: &config.Config{OpenclawConfigDir: dir, LLMConfigMode: "runtime"}}, path
}

func withRuntime(t *testing.T, r RuntimeInfo) {
	t.Helper()
	prev := openclawRuntime
	openclawRuntime = func() RuntimeInfo { return r }
	t.Cleanup(func() { openclawRuntime = prev })
}

func readConfig(t *testing.T, path string) (compaction, controlUI map[string]any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	compaction = cfg["agents"].(map[string]any)["defaults"].(map[string]any)["compaction"].(map[string]any)
	controlUI = cfg["gateway"].(map[string]any)["controlUi"].(map[string]any)
	return compaction, controlUI
}

func TestStrictRuntimeDropsRetiredKeys(t *testing.T) {
	withRuntime(t, RuntimeInfo{Year: 2026, Minor: 9, Patch: 34, Detected: true})
	s, path := writeRetiredKeysConfig(t)

	if _, err := s.ensureAgentDefaults(); err != nil {
		t.Fatal(err)
	}
	changed, err := s.ensureControlUIConfig()
	if err != nil || !changed {
		t.Fatalf("ensureControlUIConfig: changed=%v err=%v", changed, err)
	}
	compaction, controlUI := readConfig(t, path)
	if _, ok := compaction["reserveTokensFloor"]; ok {
		t.Error("reserveTokensFloor kept on a strict runtime")
	}
	if compaction["mode"] != "safeguard" {
		t.Error("compaction mode lost")
	}
	if _, ok := controlUI["allowInsecureAuth"]; ok {
		t.Error("allowInsecureAuth kept on a strict runtime")
	}

	// A second boot finds nothing to change, so the gateway is not restarted again.
	if changed, err := s.ensureControlUIConfig(); err != nil || changed {
		t.Fatalf("second ensureControlUIConfig: changed=%v err=%v", changed, err)
	}
}

func TestLegacyRuntimeKeepsTuningKeys(t *testing.T) {
	withRuntime(t, RuntimeInfo{Year: 2026, Minor: 9, Patch: 2, Detected: true})
	s, path := writeRetiredKeysConfig(t)

	if _, err := s.ensureAgentDefaults(); err != nil {
		t.Fatal(err)
	}
	if changed, err := s.ensureControlUIConfig(); err != nil || changed {
		t.Fatalf("ensureControlUIConfig: changed=%v err=%v", changed, err)
	}
	compaction, controlUI := readConfig(t, path)
	if compaction["reserveTokensFloor"] != float64(5000) {
		t.Errorf("reserveTokensFloor = %v, want 5000", compaction["reserveTokensFloor"])
	}
	if controlUI["allowInsecureAuth"] != false {
		t.Errorf("allowInsecureAuth = %v, want false", controlUI["allowInsecureAuth"])
	}
}
