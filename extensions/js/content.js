// js/content.js
// ==============================================
// 1) 유튜브 영상 재생 감지 → background.js로 pageUrl 전송
// 2) background.js → 분석 완료 메시지 수신 → 화면 오버레이 표시
// ==============================================

(() => {
  // ==============================================
  // 전역 설정
  // ==============================================
  const POLL_INTERVAL_MS = 2000;
  let lastSentUrl = null;

  // ==============================================
  // YouTube watch 페이지 여부 확인
  // ==============================================
  function isOnWatchPage() {
    return window.location.hostname.includes("youtube.com") && window.location.pathname.startsWith("/watch");
  }

  // ==============================================
  // 영상 재생 상태 확인
  // ==============================================
  function isVideoPlaying() {
    const videoEl = document.querySelector("video.html5-main-video");
    return videoEl && !videoEl.paused && !videoEl.ended && videoEl.currentTime > 0;
  }

  // ==============================================
  // 현재 페이지 URL 조회
  // ==============================================
  function getCurrentPageUrl() {
    return window.location.href;
  }

  // ==============================================
  // background.js로 영상 URL 전달
  // ==============================================
  function sendPageUrlToBackground(pageUrl) {
    chrome.runtime.sendMessage({ type: "MAIN_VIDEO_URL", payload: { pageUrl } }, (response) => {
      console.log("[AdRisk CS] background response:", response);
    });
  }

  // ==============================================
  // 유튜브 영상 변화 감시 루프
  // ==============================================
  function startVideoWatcher() {
    setInterval(() => {
      if (!isOnWatchPage()) return;
      if (!isVideoPlaying()) return;

      const pageUrl = getCurrentPageUrl();
      if (pageUrl === lastSentUrl) return;

      // 새로운 URL 감지 시 → background.js로 요청 전송
      lastSentUrl = pageUrl;
      sendPageUrlToBackground(pageUrl);
    }, POLL_INTERVAL_MS);
  }

  // ==============================================
  // 초기 로직 실행
  // ==============================================
  function init() {
    if (!isOnWatchPage()) return;

    console.log("[AdRisk CS] content.js initialized");
    startVideoWatcher();
  }

  // SPA 로딩 환경 보정
  window.addEventListener("load", () => setTimeout(init, 1500));

  // ==============================================
  // background.js → 결과 수신 이벤트
  // ==============================================
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "ADRISK_RESULT_READY") {
      renderAdRiskOverlay(message.payload);
    }
  });

  // ==============================================
  // 분석 결과 팝업 렌더링 (오버레이)
  // ==============================================
  function renderAdRiskOverlay(result) {
    const { label, final_score, video_score, text_score } = result;
    const percent = Math.round((final_score || 0) * 100);

    // ----------------------------------------------
    // 팝업 DOM 생성 (없으면 생성, 있으면 재사용)
    // ----------------------------------------------
    let box = document.getElementById("adrisk-overlay");
    if (!box) {
      box = document.createElement("div");
      box.id = "adrisk-overlay";

      box.innerHTML = `
        <div class="adrisk-card">
          <div class="adrisk-header">
            <span class="adrisk-icon" id="adriskIcon"></span>
            <span class="adrisk-title" id="adriskTitle"></span>
            <span class="adrisk-close" id="adriskClose">✕</span>
          </div>
          <div class="adrisk-subtitle">현재 시청 중인 콘텐츠 분석 결과</div>
          <div class="adrisk-body">
            <div class="adrisk-badge">
              <span id="adriskBadgeIcon"></span>
              <span id="adriskBadgeText"></span>
            </div>
            <div class="adrisk-progress-row">
              <div class="adrisk-progress-wrap">
                <div class="adrisk-progress-fill" id="adriskProgress"></div>
              </div>
              <div class="adrisk-progress-percent" id="adriskPercent"></div>
            </div>
            <div class="adrisk-detail" id="adriskDetail"></div>
          </div>
        </div>
      `;

      // ----------------------------------------------
      // 오버레이 전용 스타일 삽입
      // ----------------------------------------------
      const style = document.createElement("style");
      style.textContent = `
        #adrisk-overlay {
          position: absolute;
          top: 20px;
          right: 20px;
          z-index: 999999;
        }

        .adrisk-card {
          width: 260px;
          border-radius: 18px;
          overflow: hidden;
          background: #fff;
          box-shadow: 0 6px 16px rgba(0,0,0,0.25);
          font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo",
            "Noto Sans KR", system-ui, sans-serif;
        }

        .adrisk-header {
          display:flex; align-items:center;
          padding: 10px 12px;
          color:#fff;
          font-weight:700;
        }

        .adrisk-subtitle {
          font-size: 11px;
          padding: 4px 12px 8px;
          color:#fff;
          opacity:0.8;
        }

        .adrisk-close {
          margin-left:auto;
          cursor:pointer;
        }

        .adrisk-body {
          padding: 12px;
          background:#f5f6fa;
        }

        .adrisk-progress-row {
          display:flex; align-items:center; gap:8px;
          margin:8px 0;
        }

        .adrisk-progress-wrap {
          flex:1;
          height:10px;
          background:#eee;
          border-radius:999px;
          overflow:hidden;
        }

        .adrisk-progress-fill {
          height:100%;
          width:0%;
          background:#4caf50;
          transition: width .4s ease;
        }
        
        .adrisk-progress-percent {
          min-width: 36px;
          font-size: 12px;
          font-weight: 600;
          text-align: right;
        }
      `;
      box.appendChild(style);

      document.body.appendChild(box);

      // 닫기 버튼
      box.querySelector("#adriskClose").onclick = () => box.remove();
    }

    // ----------------------------------------------
    // label(안전/주의/위험/매우위험) → UI 매핑
    // ----------------------------------------------
    const header = box.querySelector(".adrisk-header");
    const subtitle = box.querySelector(".adrisk-subtitle");
    const icon = box.querySelector("#adriskIcon");
    const title = box.querySelector("#adriskTitle");
    const badgeIcon = box.querySelector("#adriskBadgeIcon");
    const badgeText = box.querySelector("#adriskBadgeText");
    const detail = box.querySelector("#adriskDetail");
    const progress = box.querySelector("#adriskProgress");
    const percentText = box.querySelector("#adriskPercent");

    // ----------------------------------------------
    // label UI 적용 헬퍼
    // ----------------------------------------------
    function apply(color, iconTxt, titleTxt, badgeTxt, detailTxt) {
      header.style.background = color;
      subtitle.style.background = color;
      icon.textContent = iconTxt;
      title.textContent = titleTxt;
      badgeIcon.textContent = iconTxt;
      badgeText.textContent = badgeTxt;
      detail.textContent = detailTxt;

      if (progress) progress.style.background = color;
      if (percentText) percentText.style.color = color;
    }

    // ----------------------------------------------
    // 등급별 UI 매핑
    // ----------------------------------------------
    if (label === "안전") {
      apply("#0f9d58", "✅", "안전한 광고 감지됨", "광고 신뢰도", "위험 요소가 감지되지 않았습니다.");
    } else if (label === "주의") {
      apply("#f6a623", "⚠️", "주의 요망 광고 감지됨", "광고 신뢰도", "딥페이크 위험 감지됨");
    } else if (label === "위험") {
      apply("#f46b2b", "⚠️", "위험 광고 감지됨", "광고 신뢰도", "과장광고 위험 감지됨");
    } else if (label === "매우위험") {
      apply("#e53935", "🚨", "매우 위험한 광고 감지됨", "광고 신뢰도", "딥페이크·과장광고 위험 모두 감지됨");
    }

    // ----------------------------------------------
    // 진행률 표시 업데이트
    // ----------------------------------------------
    box.querySelector("#adriskProgress").style.width = `${percent}%`;
    box.querySelector("#adriskPercent").textContent = `${percent}%`;
  }
})();
