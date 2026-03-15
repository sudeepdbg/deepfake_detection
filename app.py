import os
import numpy as np
import streamlit as st
from vision_module import VideoDetector
from audio_module import AudioDetector

st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Multimodal Deepfake Detector")
st.caption("Unified AI detection · MP4 video · JPG / PNG images · WAV / MP3 audio")

DEEPFAKE_THRESHOLD  = 0.50
UNCERTAIN_THRESHOLD = 0.35

uploaded_file = st.file_uploader(
    "Upload a file to analyse",
    type=["mp4", "wav", "mp3", "jpg", "jpeg", "png"]
)


def verdict_ui(score: float, video_score: float, audio_score,
               audio_failed: bool, media_type: str):
    st.divider()
    _, col, _ = st.columns([1, 2, 1])
    with col:
        pct = f"{score:.0%}"
        if score >= DEEPFAKE_THRESHOLD:
            label = "🤖 AI GENERATED" if media_type == "image" else "🚨 DEEPFAKE DETECTED"
            st.error(f"## {label}\nSuspicion score: **{pct}**")
        elif score >= UNCERTAIN_THRESHOLD:
            st.warning(f"## ⚠️ INCONCLUSIVE\nSuspicion score: **{pct}** — manual review recommended")
        else:
            st.success(f"## ✅ LIKELY AUTHENTIC\nSuspicion score: **{pct}**")

        if audio_failed and media_type != "image":
            st.caption("⚠️ Audio analysis unavailable — verdict based on video only")

        with st.expander("ℹ️ Score breakdown"):
            if media_type == "image":
                st.caption(f"**Image score:** {video_score:.2f}")
            else:
                st.caption(
                    f"**Video:** {video_score:.2f} · "
                    f"**Audio:** {'N/A' if audio_failed else f'{audio_score:.2f}'}  \n"
                    f"Final = max(video, audio). "
                    f"≥{DEEPFAKE_THRESHOLD:.0%} = Fake · ≥{UNCERTAIN_THRESHOLD:.0%} = Inconclusive"
                )


def score_bar(label: str, value: float):
    c1, c2, c3 = st.columns([2, 4, 1])
    c1.caption(label)
    c2.progress(float(np.clip(value, 0.0, 1.0)))
    icon = "🔴" if value > 0.5 else "🟡" if value > 0.35 else "🟢"
    c3.caption(f"{icon} {value:.2f}")


def show_vision_signals(r: dict):
    media = r.get("_media_type", "unknown")
    vw    = r.get("_video_w", 0)
    iw    = r.get("_image_w", 0)
    st.caption(f"*Detected as: **{media}** · image_weight={iw:.2f}, video_weight={vw:.2f}*")
    score_bar("Chroma noise correlation",    r.get("chroma_noise_corr", 0))
    score_bar("Noise autocorrelation (PRNU)", r.get("noise_autocorr",   0))
    score_bar("Skin texture smoothness",     r.get("skin_smooth",       0))
    score_bar("ELA uniformity",              r.get("ela_uniformity",    0))
    score_bar("Edge coherence (Laplacian)",  r.get("edge_coherence",    0))
    score_bar("GAN noise fingerprint",       r.get("noise_residual",    0))
    score_bar("Over-sharpening artefacts",   r.get("over_sharpening",   0))
    score_bar("Face/background separation",  r.get("bg_separation",     0))


if uploaded_file is not None:
    ext      = os.path.splitext(uploaded_file.name)[1].lower()
    tmp_path = f"temp_upload{ext}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.info(f"📁 **{uploaded_file.name}** — {uploaded_file.size / 1024:.1f} KB")

    if st.button("🔍 Run Deepfake Analysis", type="primary", use_container_width=True):

        video_score  = 0.0
        audio_score  = 0.0
        audio_failed = False
        is_image     = ext in (".jpg", ".jpeg", ".png")

        # ── STILL IMAGE ───────────────────────────────────────────────────
        if is_image:
            with st.spinner("Analysing image…"):
                result = VideoDetector().analyze_image_file(tmp_path)
            if "error" in result and result["score"] == 0.0:
                st.error(f"Analysis failed: {result['error']}")
            else:
                video_score = result["score"]
                st.metric("AI Suspicion Score", f"{video_score:.2f}")
                show_vision_signals(result)
            audio_failed = True

        # ── MP4 VIDEO ─────────────────────────────────────────────────────
        elif ext == ".mp4":
            col_v, col_a = st.columns(2)

            with col_v:
                st.markdown("#### 🎞️ Frame-Level Analysis")
                with st.spinner("Decoding and analysing frames…"):
                    vr = VideoDetector().analyze_video_file(tmp_path, n_frames=30)
                if vr.get("frames_analysed", 0) == 0:
                    st.error(f"Frame analysis failed: {vr.get('error', 'unknown')}")
                else:
                    video_score = vr["score"]
                    st.metric("Video Score", f"{video_score:.2f}",
                              delta=f"{vr['frames_analysed']}/{vr['total_frames']} frames",
                              delta_color="off")
                    show_vision_signals(vr)
                    score_bar("Temporal inconsistency", vr.get("temporal", 0))
                    if vr.get("frame_scores"):
                        import pandas as pd
                        st.caption("Frame-by-frame suspicion:")
                        st.bar_chart(pd.DataFrame({"suspicion": vr["frame_scores"]}), height=130)

            with col_a:
                st.markdown("#### 🔊 Audio Track Analysis")
                with st.spinner("Extracting and analysing audio…"):
                    ar = AudioDetector().predict_audio_file(tmp_path)
                if "error" in ar:
                    st.warning(f"Audio: {ar['error']}")
                    audio_failed = True
                else:
                    audio_score = ar["score"]
                    st.metric("Audio Score", f"{audio_score:.2f}",
                              delta=f"{ar.get('_duration_s','?')}s", delta_color="off")
                    score_bar("Spectral flux",    ar.get("spec_flux",   0))
                    score_bar("Energy variation", ar.get("energy_var",  0))
                    score_bar("ZCR consistency",  ar.get("zcr_consist", 0))
                    score_bar("Spectral rolloff", ar.get("rolloff",     0))
                    score_bar("Silence pattern",  ar.get("silence",     0))
                    with st.expander("📐 Raw audio diagnostics"):
                        st.json({k: v for k, v in ar.items() if k.startswith("_")})

        # ── AUDIO ONLY ────────────────────────────────────────────────────
        elif ext in (".wav", ".mp3"):
            with st.spinner("Analysing audio…"):
                ar = AudioDetector().predict_audio_file(tmp_path)
            if "error" in ar:
                st.error(f"Audio analysis failed: {ar['error']}")
                audio_failed = True
            else:
                audio_score = ar["score"]
                st.metric("Audio Suspicion Score", f"{audio_score:.2f}")
                score_bar("Spectral flux",    ar.get("spec_flux",   0))
                score_bar("Energy variation", ar.get("energy_var",  0))
                score_bar("ZCR consistency",  ar.get("zcr_consist", 0))
                score_bar("Spectral rolloff", ar.get("rolloff",     0))
                score_bar("Silence pattern",  ar.get("silence",     0))

        # ── Verdict ───────────────────────────────────────────────────────
        verdict_ui(max(video_score, audio_score), video_score, audio_score,
                   audio_failed, "image" if is_image else "video")
