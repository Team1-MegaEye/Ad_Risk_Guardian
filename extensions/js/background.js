// js/background.js
// ==============================================
// FastAPI 백엔드(/predict, /result)와 통신하고
// 분석 결과를 해당 탭(content.js)로 전달하는 서비스 워커
// ==============================================

// ==============================================
// 백엔드 API 엔드포인트 설정
// ==============================================
const BACKEND_BASE_URL = "http://127.0.0.1:8000";

// POST /predict : YouTube URL을 받아 비동기 분석 태스크(run_inference) 생성, task_id 반환
const PREDICT_ENDPOINT = `${BACKEND_BASE_URL}/predict`;

// GET  /result/:task_id : Celery 작업 상태 및 최종 결과 조회
const RESULT_ENDPOINT = `${BACKEND_BASE_URL}/result`;

const ENABLE_BACKEND_CALL = true;

// ==============================================
// taskId → tabId 매핑 저장
// ==============================================
const taskTabMap = {};

// ==============================================
// 확장 프로그램 설치/업데이트 이벤트
// ==============================================
chrome.runtime.onInstalled.addListener(() => {
  console.log("[AdRisk BG] background.js loaded (MV3)");
});

// ==============================================
// content.js → 메시지 수신
// type: MAIN_VIDEO_URL
// ==============================================
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "MAIN_VIDEO_URL") return;

  const pageUrl = message.payload?.pageUrl;
  const tabId = sender.tab?.id;

  // ----------------------------------------------
  // URL 누락 시 에러 반환
  // ----------------------------------------------
  if (!pageUrl) {
    sendResponse({ ok: false, error: "pageUrl missing" });
    return;
  }

  console.log("[AdRisk BG] 받은 pageUrl:", pageUrl, "탭 ID:", tabId);

  // ----------------------------------------------
  // 개발 중 API 호출 비활성화 모드
  // ----------------------------------------------
  if (!ENABLE_BACKEND_CALL) {
    sendResponse({ ok: true, forwarded: false });
    return;
  }

  // ==============================================
  // /predict 전송 (비동기 처리)
  // ==============================================
  (async () => {
    try {
      const res = await fetch(PREDICT_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: pageUrl }),
      });

      if (!res.ok) {
        sendResponse({ ok: false, forwarded: true });
        return;
      }

      const data = await res.json();
      console.log("[AdRisk BG] /predict 응답:", data);

      const taskId = data.task_id ?? null;

      // ----------------------------------------------
      // taskId → tabId 매핑 저장 + Polling 시작
      // ----------------------------------------------
      if (taskId && tabId != null) {
        taskTabMap[taskId] = tabId;
        pollResult(taskId);
      }

      sendResponse({ ok: true, forwarded: true, taskId });
    } catch (err) {
      console.error("[AdRisk BG] fetch 에러:", err);
      sendResponse({ ok: false, forwarded: true, error: String(err) });
    }
  })();

  return true; // keep sendResponse alive
});

// ==============================================
// /result/{taskId} Polling
// 완료되면 content.js로 메시지 전달
// ==============================================
async function pollResult(taskId, interval = 3000, maxTry = 20) {
  console.log("[AdRisk BG] Poll /result for:", taskId);

  for (let attempt = 1; attempt <= maxTry; attempt++) {
    try {
      const res = await fetch(`${RESULT_ENDPOINT}/${taskId}`);
      if (!res.ok) break;

      const data = await res.json();
      console.log("[AdRisk BG] /result 응답:", data);

      // ----------------------------------------------
      // 아직 처리 중이면 재시도
      // ----------------------------------------------
      if (data.status === "processing") {
        await new Promise((r) => setTimeout(r, interval));
        continue;
      }

      // ----------------------------------------------
      // 완료된 경우 → Popup + content.js 업데이트
      // ----------------------------------------------
      if (data.status === "completed") {
        const result = {
          ...data.result,
          taskId,
          updatedAt: Date.now(),
        };

        // popup(확장 프로그램 UI)에 표시할 저장값 갱신
        chrome.storage.local.set({ lastAdRiskResult: result });

        // 해당 광고가 있던 탭에 메시지 보내기
        const tabId = taskTabMap[taskId];
        if (tabId != null) {
          chrome.tabs.sendMessage(tabId, {
            type: "ADRISK_RESULT_READY",
            payload: result,
          });
        }
        return;
      }
    } catch (err) {
      console.error("[AdRisk BG] result fetch error:", err);
      break;
    }
  }

  console.warn("[AdRisk BG] Polling timeout:", taskId);
}
