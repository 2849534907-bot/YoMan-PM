# -*- coding: utf-8 -*-
"""前端补丁：给 index.html 加文件上传功能（一次性脚本）。"""
import io

P = r"C:\Users\Administrator\Desktop\Agent开发\EchoMind-PM\web\index.html"
with io.open(P, encoding="utf-8") as f:
    t = f.read()
t = t.replace("\r\n", "\n")

reps = []

# 1) CSS：文件上传样式
old_css = "</style>"
new_css = """  /* 文件上传 */
  .file-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
  .file-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 4px 12px; font-size: 12px; color: var(--text);
    box-shadow: var(--shadow); max-width: 260px;
  }
  .file-chip .chip-icon { font-size: 13px; }
  .file-chip .chip-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .file-chip .chip-note { color: var(--warning); font-size: 11px; }
  .file-chip .chip-x { cursor: pointer; color: var(--text-secondary); font-weight: 700; padding: 0 2px; }
  .file-chip .chip-x:hover { color: #ef4444; }
  .file-chip.uploading { opacity: .6; }
  .upload-btn {
    background: transparent; border: none; cursor: pointer; font-size: 20px;
    color: var(--text-secondary); padding: 4px 10px; border-radius: 8px; flex-shrink: 0;
  }
  .upload-btn:hover { background: var(--agent-bubble); color: var(--primary); }
</style>"""
reps.append((old_css, new_css))

# 2) HTML：input-area 加上传按钮和附件容器
old_html = """<div class="input-area">
  <div class="input-wrapper">
    <textarea class="input-box" id="messageInput" placeholder="输入你的问题，按 Enter 发送，Shift+Enter 换行..." rows="1"></textarea>
    <button class="send-btn" id="sendBtn" onclick="sendMessage()">
      <span>发送</span>
    </button>
  </div>
  <div class="footer-hint">API: <span id="apiUrl">http://127.0.0.1:8123</span> · 豆包 doubao-seed-evolving 驱动</div>
</div>"""
new_html = """<div class="input-area">
  <div class="file-chips" id="fileChips"></div>
  <div class="input-wrapper">
    <button class="upload-btn" id="uploadBtn" title="上传文件（文档/表格/PPT/PDF/图片/视频等，最多5个）" onclick="document.getElementById('fileInput').click()">📎</button>
    <textarea class="input-box" id="messageInput" placeholder="输入你的问题，按 Enter 发送，Shift+Enter 换行..." rows="1"></textarea>
    <button class="send-btn" id="sendBtn" onclick="sendMessage()">
      <span>发送</span>
    </button>
  </div>
  <input type="file" id="fileInput" multiple style="display:none" />
  <div class="footer-hint">支持 Word / Excel / PPT / PDF / 图片 / 文本 / 压缩包等 · API: <span id="apiUrl">http://127.0.0.1:8123</span></div>
</div>"""
reps.append((old_html, new_html))

# 3) sendMessage 开头：附件随消息发送
old_send = """  async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isLoading) return;

    addMessage('user', text);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    setLoading(true);"""
new_send = """  async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text && pendingFiles.length === 0) return;
    if (isLoading) return;

    // 附件随消息发送，发送后清空
    const payloadFiles = pendingFiles.map(f => ({ filename: f.filename, kind: f.kind, content: f.content, note: f.note }));
    const attachNames = pendingFiles.map(f => f.filename);
    const userText = attachNames.length ? (text + '\\n📎 ' + attachNames.join('、')) : text;
    addMessage('user', userText);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    pendingFiles = [];
    renderFileChips();
    setLoading(true);"""
reps.append((old_send, new_send))

# 4) body 加 files
old_body = """        body: JSON.stringify({
          user_id: userId,
          session_id: sessionId,
          message: text,
          model: getModelForRequest()
        }),"""
new_body = """        body: JSON.stringify({
          user_id: userId,
          session_id: sessionId,
          message: text,
          model: getModelForRequest(),
          files: payloadFiles
        }),"""
reps.append((old_body, new_body))

# 5) 文件上传 JS（插在 sendMessage 前）
anchor = "  async function sendMessage() {"
new_js = """  // ── 文件上传 ──────────────────────────────────────────────────────────────
  let pendingFiles = [];
  const fileInput = document.getElementById('fileInput');
  const fileChips = document.getElementById('fileChips');

  function fileIcon(name) {
    const n = name.toLowerCase();
    if (/\\.(png|jpe?g|gif|webp|bmp)$/.test(n)) return '🖼️';
    if (/\\.(mp4|avi|mkv|mov|wmv|flv|webm)$/.test(n)) return '🎬';
    if (/\\.(mp3|wav|m4a|aac|ogg)$/.test(n)) return '🎵';
    if (/\\.(docx?|doc)$/.test(n)) return '📄';
    if (/\\.(xlsx?|xls|csv)$/.test(n)) return '📊';
    if (/\\.(pptx?|ppt)$/.test(n)) return '📽️';
    if (/\\.pdf$/.test(n)) return '📕';
    if (/\\.zip$/.test(n)) return '🗜️';
    return '📎';
  }

  function renderFileChips() {
    fileChips.innerHTML = '';
    pendingFiles.forEach((f, i) => {
      const chip = document.createElement('span');
      chip.className = 'file-chip';
      chip.innerHTML = '<span class="chip-icon">' + fileIcon(f.filename) + '</span>' +
        '<span class="chip-name" title="' + f.filename + '">' + f.filename + '</span>' +
        (f.note ? '<span class="chip-note" title="' + f.note + '">⚠</span>' : '') +
        '<span class="chip-x" onclick="removeFile(' + i + ')">✕</span>';
      fileChips.appendChild(chip);
    });
  }

  function removeFile(i) {
    pendingFiles.splice(i, 1);
    renderFileChips();
  }

  async function handleFileSelect(e) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    for (const f of files) {
      if (pendingFiles.length >= 5) { alert('最多同时上传 5 个文件'); break; }
      const chip = document.createElement('span');
      chip.className = 'file-chip uploading';
      chip.innerHTML = '<span class="chip-icon">' + fileIcon(f.name) + '</span><span class="chip-name">' + f.name + '</span><span style="font-size:11px">解析中…</span>';
      fileChips.appendChild(chip);
      try {
        const fd = new FormData();
        fd.append('file', f);
        const r = await fetch(API_BASE + '/upload-file', { method: 'POST', body: fd });
        const res = await r.json();
        if (!r.ok) throw new Error(res.detail || '上传失败');
        pendingFiles.push({ filename: res.filename, kind: res.kind, content: res.content, note: res.note || '' });
      } catch (err) {
        alert('文件「' + f.name + '」上传失败：' + err.message);
      }
    }
    renderFileChips();
    e.target.value = '';
  }
  fileInput.addEventListener('change', handleFileSelect);

""" + anchor
reps.append((anchor, new_js))

for i, (old, new) in enumerate(reps, 1):
    cnt = t.count(old)
    if cnt == 0:
        print(f"[{i}] MISS: {old[:60]!r}")
    else:
        t = t.replace(old, new)
        print(f"[{i}] OK (x{cnt}): {old[:40]!r}")

t = t.replace("\n", "\r\n")
with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(t)
print("写入完成")
