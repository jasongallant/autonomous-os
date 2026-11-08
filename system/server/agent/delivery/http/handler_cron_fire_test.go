package http

import "testing"

// Payload shapes logged by OpenClaw 2026.9.34 (`cron event raw payload`).
const (
	declaredCronStarted = `{"jobId":"c5eeb90a","action":"started","job":{"id":"c5eeb90a","declarationKey":"memory-core:memory-dreaming-promotion","name":"Memory Dreaming Promotion"}}`
	userCronStarted     = `{"jobId":"9f1e2d3c","action":"started","job":{"id":"9f1e2d3c","name":"Remind Jason to stretch"}}`
)

func TestParseCronStarted(t *testing.T) {
	slot, ok := parseCronStarted([]byte(declaredCronStarted), 1000)
	if !ok || slot.jobID != "c5eeb90a" || !slot.silent() || slot.at != 1000 {
		t.Fatalf("declared job: ok=%v slot=%+v", ok, slot)
	}
	slot, ok = parseCronStarted([]byte(userCronStarted), 1000)
	if !ok || slot.silent() {
		t.Fatalf("user job: ok=%v slot=%+v", ok, slot)
	}
	if _, ok := parseCronStarted([]byte(`{"jobId":"x","action":"finished"}`), 1000); ok {
		t.Fatal("a non-started action queued a slot")
	}
	if _, ok := parseCronStarted([]byte(`not json`), 1000); ok {
		t.Fatal("bad JSON queued a slot")
	}
}

// A plugin-declared job's run is kept off the speaker; a user job's run is still forced onto it.
func TestClaimCronFireMarksDeclaredJobsSilent(t *testing.T) {
	h := &AgentHandler{}
	declared, _ := parseCronStarted([]byte(declaredCronStarted), 1000)
	user, _ := parseCronStarted([]byte(userCronStarted), 1500)
	h.expectCronFire(declared)
	h.expectCronFire(user)

	if slot, ok := h.claimCronFire("run-a", 2000); !ok || !slot.silent() {
		t.Fatalf("first claim: ok=%v slot=%+v", ok, slot)
	}
	if slot, ok := h.claimCronFire("run-b", 2100); !ok || slot.silent() {
		t.Fatalf("second claim: ok=%v slot=%+v", ok, slot)
	}
	if !h.silentCronRuns["run-a"] || h.cronFireRuns["run-a"] {
		t.Error("declared job's run was not marked silent")
	}
	if !h.cronFireRuns["run-b"] || h.silentCronRuns["run-b"] {
		t.Error("user job's run was not forced to the speaker")
	}
	if _, ok := h.claimCronFire("run-c", 2200); ok {
		t.Error("a run with no queued cron claimed a slot")
	}
}

func TestClaimCronFireIgnoresExpiredSlots(t *testing.T) {
	h := &AgentHandler{}
	user, _ := parseCronStarted([]byte(userCronStarted), 1000)
	h.expectCronFire(user)
	if _, ok := h.claimCronFire("late-run", 1000+cronFireWindowMs+1); ok {
		t.Fatal("a slot older than cronFireWindowMs was claimed")
	}
	if len(h.cronFireExpected) != 0 {
		t.Fatalf("expired slot kept: %d", len(h.cronFireExpected))
	}
}
