document.getElementById('scanBtn').addEventListener('click', async () => {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = "<em>AI is analyzing... 🧠</em>";

    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => window.getSelection().toString()
    }, async (injectionResults) => {
        const selectedText = injectionResults[0].result;

        if (!selectedText || selectedText.trim() === "") {
            resultDiv.innerHTML = "⚠️ <strong>Wait!</strong> Please highlight a sentence first.";
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:8000/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: selectedText })
            });

            const data = await response.json();

            if (data.findings && data.findings.length > 0) {
                resultDiv.innerHTML = "<strong>Analysis Results:</strong>";
                data.findings.forEach(item => {
                    // Risk level ke hisaab se color decide karein
                    const badgeColor = item.risk_level === "High" ? "#e74c3c" : "#f1c40f";
                    const textColor = item.risk_level === "High" ? "white" : "black";

                    resultDiv.innerHTML += `
                        <div style="border-left: 5px solid ${badgeColor}; background: #fff; padding: 10px; margin-top: 10px; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <span style="background: ${badgeColor}; color: ${textColor}; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; text-transform: uppercase;">
                                ${item.risk_level} RISK
                            </span>
                            <div style="margin-top: 5px; font-weight: 600; color: #333;">🚩 ${item.category}</div>
                            <div style="font-size: 11px; color: #666;">Confidence: ${(item.probability * 100).toFixed(1)}%</div>
                        </div>`;
                });
            } else {
                resultDiv.innerHTML = "<div style='color: #27ae60; font-weight: bold; text-align: center; padding: 20px;'>✅ This clause looks safe!</div>";
            }
        } catch (error) {
            resultDiv.innerHTML = "❌ <strong>Connection Error:</strong> Is your Python server running on port 8000?";
        }
    });
});