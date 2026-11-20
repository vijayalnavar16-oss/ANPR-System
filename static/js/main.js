// ANPR Dashboard Frontend Controller

document.addEventListener('DOMContentLoaded', () => {
    initControls();
    initPolling();
    initFilters();
});

function selectQuickSource(src) {
    const input = document.getElementById('cameraSourceInput');
    if (input) {
        input.value = src;
        document.getElementById('btnApplySource')?.click();
    }
}

// 1. Controls & Camera Buttons
function initControls() {
    const btnStart = document.getElementById('btnStartCam');
    const btnStop = document.getElementById('btnStopCam');
    const btnApplySource = document.getElementById('btnApplySource');
    const sourceInput = document.getElementById('cameraSourceInput');

    if (btnStart) {
        btnStart.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/camera/start', { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    btnStart.style.display = 'none';
                    btnStop.style.display = 'inline-flex';
                    updateCameraStatusPill(true);
                    reloadStream();
                } else {
                    alert('Camera error: ' + (data.error || 'Failed to start camera.'));
                }
            } catch (err) {
                console.error(err);
            }
        });
    }

    if (btnStop) {
        btnStop.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/camera/stop', { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    btnStop.style.display = 'none';
                    btnStart.style.display = 'inline-flex';
                    updateCameraStatusPill(false);
                    reloadStream();
                }
            } catch (err) {
                console.error(err);
            }
        });
    }

    if (btnApplySource) {
        btnApplySource.addEventListener('click', async () => {
            const newSource = sourceInput.value.trim();
            try {
                const res = await fetch('/api/camera/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: newSource })
                });
                const data = await res.json();
                if (data.success) {
                    reloadStream();
                    console.log(`Camera source switched to: ${newSource}`);
                }
            } catch (err) {
                console.error(err);
            }
        });
    }
}

function updateCameraStatusPill(isOnline) {
    const pill = document.getElementById('cameraStatusPill');
    const text = document.getElementById('cameraStatusText');
    if (!pill || !text) return;

    if (isOnline) {
        pill.className = 'status-pill online';
        text.textContent = 'Camera Active';
    } else {
        pill.className = 'status-pill offline';
        text.textContent = 'Camera Inactive';
    }
}

function reloadStream() {
    const img = document.getElementById('liveStream');
    if (img) {
        img.src = '/video_feed?t=' + new Date().getTime();
    }
}

function handleStreamError(img) {
    console.warn("Video stream offline or waiting for frames, retrying in 2s...");
    setTimeout(() => {
        if (img) {
            img.src = '/video_feed?t=' + new Date().getTime();
        }
    }, 2000);
}

// 2. Real-time Polling for Stats & Latest Detection
function initPolling() {
    // Poll stats every 3 seconds
    setInterval(fetchStats, 3000);
    // Poll latest detection every 1.5 seconds
    setInterval(fetchLatestDetection, 1500);
    // Poll logs every 4 seconds
    setInterval(() => {
        // Only auto-poll if user is not actively searching
        const searchInput = document.getElementById('filterSearch');
        if (searchInput && !searchInput.value.trim()) {
            fetchFilteredRecords();
        }
    }, 4000);
}

async function fetchStats() {
    try {
        const res = await fetch('/api/stats');
        if (!res.ok) return;
        const stats = await res.json();

        document.getElementById('statTotal').textContent = stats.total_detected;
        document.getElementById('statToday').textContent = stats.today_entries;
        document.getElementById('statValid').textContent = stats.valid_plates;
        document.getElementById('statPending').textContent = stats.pending_verification;
        const rejEl = document.getElementById('statRejected');
        if (rejEl) rejEl.textContent = stats.rejected_detections ?? 0;
    } catch (err) {
        console.error("Failed to fetch stats:", err);
    }
}

async function fetchLatestDetection() {
    try {
        const res = await fetch('/api/latest_detection');
        if (!res.ok) return;
        const det = await res.json();

        const plateElem = document.getElementById('latestPlateNumber');
        const rawElem = document.getElementById('latestRawText');
        const confElem = document.getElementById('latestConfidence');
        const timeElem = document.getElementById('latestTimestamp');
        const statusBadge = document.getElementById('latestStatusBadge');

        if (plateElem && det.plate_number) plateElem.textContent = det.plate_number;
        if (rawElem && det.raw_text) rawElem.textContent = det.raw_text;
        if (confElem && det.confidence) confElem.textContent = `${det.confidence}%`;
        if (timeElem && det.timestamp) timeElem.textContent = det.timestamp;

        if (statusBadge && det.verification_status) {
            statusBadge.textContent = det.verification_status;
            statusBadge.className = `status-badge status-${det.verification_status.toLowerCase()}`;
        }
    } catch (err) {
        console.error("Failed to fetch latest detection:", err);
    }
}

// 3. Search & Filtering
function initFilters() {
    const search = document.getElementById('filterSearch');
    const date = document.getElementById('filterDate');
    const status = document.getElementById('filterStatus');
    const btnReset = document.getElementById('btnResetFilters');
    const btnRefresh = document.getElementById('btnRefreshLogs');

    if (search) search.addEventListener('input', debounce(fetchFilteredRecords, 300));
    if (date) date.addEventListener('change', fetchFilteredRecords);
    if (status) status.addEventListener('change', fetchFilteredRecords);

    if (btnReset) {
        btnReset.addEventListener('click', () => {
            if (search) search.value = '';
            if (date) date.value = '';
            if (status) status.value = 'ALL';
            fetchFilteredRecords();
        });
    }

    if (btnRefresh) {
        btnRefresh.addEventListener('click', fetchFilteredRecords);
    }
}

async function fetchFilteredRecords() {
    const search = document.getElementById('filterSearch')?.value || '';
    const date = document.getElementById('filterDate')?.value || '';
    const status = document.getElementById('filterStatus')?.value || '';

    const url = `/api/records?search=${encodeURIComponent(search)}&date=${encodeURIComponent(date)}&status=${encodeURIComponent(status)}&limit=40`;

    try {
        const res = await fetch(url);
        if (!res.ok) return;
        const records = await res.json();
        renderTable(records);
    } catch (err) {
        console.error("Failed to filter records:", err);
    }
}

function friendlyStatus(s) {
    const map = {
        'VALID_INDIAN_PLATE': 'Valid',
        'PENDING_RECHECK': 'Pending',
        'REJECTED_INVALID_FORMAT': 'Invalid Fmt',
        'REJECTED_WATERMARK': 'Watermark',
        'UNREADABLE': 'Unreadable',
        'MANUAL_VERIFICATION': 'Review',
        'CONFIRMED': 'Confirmed',
        'REJECTED': 'Rejected'
    };
    return map[s] || s;
}

function renderTable(records) {
    const tbody = document.getElementById('recordsBody');
    if (!tbody) return;

    if (!records || records.length === 0) {
        tbody.innerHTML = '<tr id="emptyRow"><td colspan="8" class="text-center text-muted">No matching vehicle records found.</td></tr>';
        return;
    }

    tbody.innerHTML = records.map(rec => {
        const confPct = (rec.confidence * 100).toFixed(1);
        const detConfPct = rec.detection_confidence ? (rec.detection_confidence * 100).toFixed(0) + '%' : '—';
        const regType = rec.registration_type || '—';
        const frameCount = rec.frame_count ? `${rec.frame_count}f` : '—';
        const imgHtml = rec.image_path
            ? `<img src="/image_view?path=${encodeURIComponent(rec.image_path)}" class="plate-thumb" alt="Plate" onclick="openImageModal(this.src);">`
            : '<span class="text-muted text-xs">No image</span>';

        const rawNotice = (rec.raw_text && rec.raw_text !== rec.plate_number)
            ? `<div class="text-xs text-muted">Raw: ${escapeHtml(rec.raw_text)}</div>`
            : '';

        const recJson = escapeHtml(JSON.stringify(rec));

        return `
            <tr data-id="${rec.id}">
                <td>
                    <span class="badge plate">${escapeHtml(rec.plate_number)}</span>
                    ${rawNotice}
                </td>
                <td>${imgHtml}</td>
                <td>${confPct}%</td>
                <td class="text-xs">${detConfPct}</td>
                <td class="text-xs">${escapeHtml(regType)}</td>
                <td class="text-xs">${rec.timestamp}</td>
                <td>
                    <span class="status-badge status-${rec.verification_status.toLowerCase()}">
                        ${friendlyStatus(rec.verification_status)}
                    </span>
                </td>
                <td>
                    <button class="btn btn-xs btn-review" onclick='openVerifyModal(${recJson});'>
                        <i class="fa-solid fa-pen-to-square"></i> Review
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

// 4. Manual Verification Modal
function openVerifyModal(record) {
    if (typeof record === 'string') {
        record = JSON.parse(record);
    }

    document.getElementById('modalRecordId').value = record.id;
    document.getElementById('modalRawText').value = record.raw_text || record.plate_number;
    document.getElementById('modalPlateNumber').value = record.plate_number || '';
    document.getElementById('modalNotes').value = record.manual_notes || '';

    const img = document.getElementById('modalPlateImg');
    if (img) {
        img.src = record.image_path ? `/image_view?path=${encodeURIComponent(record.image_path)}` : '';
        img.style.display = record.image_path ? 'block' : 'none';
    }

    document.getElementById('verifyModal').classList.add('active');
}

function closeVerifyModal() {
    document.getElementById('verifyModal').classList.remove('active');
}

async function submitVerification(action) {
    const recordId = document.getElementById('modalRecordId').value;
    const plateNumber = document.getElementById('modalPlateNumber').value.trim();
    const notes = document.getElementById('modalNotes').value.trim();

    try {
        const res = await fetch(`/api/records/${recordId}/verify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                plate_number: plateNumber,
                action: action,
                notes: notes
            })
        });

        const data = await res.json();
        if (data.success) {
            closeVerifyModal();
            fetchFilteredRecords();
            fetchStats();
        } else {
            alert('Verification failed: ' + (data.error || 'Server rejected update.'));
        }
    } catch (err) {
        console.error("Verification error:", err);
    }
}

// 5. Image Preview Modal
function openImageModal(src) {
    const modal = document.getElementById('imageModal');
    const img = document.getElementById('previewModalImg');
    if (modal && img) {
        img.src = src;
        modal.classList.add('active');
    }
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    if (modal) modal.classList.remove('active');
}

// Camera Rotation & Quick Source Helpers
async function setRotation(angle) {
    try {
        const res = await fetch('/api/camera/rotation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rotation: angle })
        });
        const data = await res.json();
        if (data.success) {
            reloadStream();
        }
    } catch (err) {
        console.error("Failed to set camera rotation:", err);
    }
}

function selectQuickSource(sourceVal) {
    const input = document.getElementById('cameraSourceInput');
    const btnApply = document.getElementById('btnApplySource');
    if (input) {
        input.value = sourceVal;
        if (btnApply) btnApply.click();
    }
}

// Helpers
function debounce(func, delay) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => func(...args), delay);
    };
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
