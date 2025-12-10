// js/app.js

// ==============================================
// popup.html 이 열릴 때 실행
// 최신 광고 분석 결과(lastAdRiskResult)를 popup 화면에 표시
// ==============================================
document.addEventListener("DOMContentLoaded", () => {
  // ==============================================
  // 저장된 최근 광고 위험도 결과 불러오기
  // chrome.storage.local.get("lastAdRiskResult", ...)
  // ==============================================
  chrome.storage.local.get("lastAdRiskResult", ({ lastAdRiskResult }) => {
    // ==============================================
    // DOM 요소 레퍼런스
    // ==============================================
    const status = document.getElementById("statusText");
    const v = document.getElementById("videoScore");
    const t = document.getElementById("textScore");
    const f = document.getElementById("finalScore");
    const lbl = document.getElementById("labelVal");

    // ==============================================
    // 결과가 없을 경우
    // ==============================================
    if (!lastAdRiskResult) {
      status.textContent = "아직 분석된 광고가 없습니다.";
      return;
    }

    // ==============================================
    // 저장된 결과 표시
    // ==============================================
    status.textContent = `최근 분석 결과 (${lastAdRiskResult.label})`;

    // 개별 지표를 % 형태로 변환하여 표시
    v.textContent = (lastAdRiskResult.video_score * 100).toFixed(1) + "%";
    t.textContent = (lastAdRiskResult.text_score * 100).toFixed(1) + "%";
    f.textContent = (lastAdRiskResult.final_score * 100).toFixed(1) + "%";
    lbl.textContent = lastAdRiskResult.label;
  });
});
