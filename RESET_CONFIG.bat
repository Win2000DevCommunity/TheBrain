@echo off
echo Resetting brain.conf to correct values...
set TB=%~dp0
set TB=%TB:~0,-1%

(
echo # TheBrain v13.0 config - CORRECTED for CPU inference
echo lr=0.001000
echo dropout=0.2000
echo epochs=400
echo iso_thresh_min=0.5000
echo iso_thresh_max=0.9500
echo nu=0.0500
echo conf_thresh=0.8500
echo max_checkpoints=10
echo async_ops=1
echo log_level=1
echo watchdog_ms=600000
echo t_lr_max=0.001000
echo t_lr_min=0.000100
echo t_warmup=500
echo t_total=20000
echo t_wd=0.010000
echo t_grad_clip=1.0000
echo t_batch=4
echo t_ctx=512
echo use_swiglu=0
echo use_rmsnorm=0
echo tie_embeddings=0
echo temperature=0.8000
echo top_k=10
echo cot_think=192
echo cot_answer=512
echo early_stop=5
echo conv_max_tokens=64
echo conv_use_facts=1
echo conv_stream=1
echo conv_history_turns=8
echo guard_enabled=0
echo train_use_conv=1
echo train_use_text=1
echo vocab_file=vocab.bpak
echo model_file=model_v13.bin
echo embeds_file=embeds.bin
echo corpus_file=corpus.bin
) > "%TB%\brain.conf"

echo Done! brain.conf reset:
echo   watchdog_ms    = 600000  (10 minutes, was 120 seconds)
echo   conv_max_tokens = 64     (was 512 - caused infinite loop)
echo   top_k           = 10    (was 40)
echo.
echo Start TheBrainV13.exe now.
pause
