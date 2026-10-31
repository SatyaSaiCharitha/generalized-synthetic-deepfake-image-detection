import asyncio
from pathlib import Path
from server import health_check, get_model_info, get_preset_samples, predict_image

print("=" * 60)
print("  GENIMAGE DETECTOR — SYSTEM & MODEL VALIDATION SUITE")
print("=" * 60)

# 1. Health check
print("\n[1] Testing /api/health...")
h = health_check()
print(f"Health status: {h}")
assert h["status"] == "online"
assert h["model_loaded"] is True
assert h["architecture"] == "convnext_tiny"
print("[+] Health check passed!")

# 2. Model Info / Benchmarks
print("\n[2] Testing /api/model-info...")
info = get_model_info()
selected = info["selected_model"]
print(f"Backbone: {selected['architecture']}")
print(f"Accuracy: {selected['test_accuracy']}%")
print(f"F1-Score: {selected['test_f1']}%")
print(f"ROC-AUC:  {selected['test_auc']}%")
print(f"Confusion Matrix: {info['confusion_matrix']}")
assert selected["architecture"] == "ConvNeXt-Tiny"
assert selected["test_accuracy"] == 98.86
assert selected["test_auc"] == 99.90
print("[+] Model-info verification passed!")

# 3. Preset Samples
print("\n[3] Testing /api/samples...")
samples = get_preset_samples()
print(f"Loaded {len(samples)} demonstration sample images.")
assert len(samples) >= 4
print("[+] Preset samples verified!")

# 4. End-to-End Prediction on All Presets
async def run_prediction_tests():
    print("\n[4] Running Prediction Validation (Multi-Scale Ensemble & Single Crop)...")
    for s in samples:
        sample_id = s["id"]
        expected = s["expected"]
        print(f"\n--- Testing sample: {sample_id} (Expected: {expected}) ---")
        
        # Test ensemble mode
        res_ens = await predict_image(preset_sample=sample_id, mode="ensemble", threshold=0.50)
        print(f"  [Ensemble] Verdict: {res_ens['verdict']} | Conf: {res_ens['confidence']}% | Latency: {res_ens['metadata']['inference_time_ms']}ms")
        print(f"             Probabilities: REAL={res_ens['probabilities']['REAL']}%, AI={res_ens['probabilities']['AI_GENERATED']}%")
        print(f"             Risk Level: {res_ens['risk_assessment']['level']}")
        assert res_ens["verdict"] == expected, f"Ensemble prediction mismatch for {sample_id}"
        assert res_ens["metadata"]["architecture"] == "convnext_tiny"
        assert res_ens["artifact_heatmap"].startswith("data:image/png;base64,")

        # Test single mode
        res_single = await predict_image(preset_sample=sample_id, mode="single", threshold=0.50)
        print(f"  [Single]   Verdict: {res_single['verdict']} | Conf: {res_single['confidence']}%")
        assert res_single["verdict"] == expected, f"Single prediction mismatch for {sample_id}"

asyncio.run(run_prediction_tests())

print("\n" + "=" * 60)
print("  ALL SYSTEM CHECKS AND MODEL INFERENCE TESTS PASSED!")
print("=" * 60)
