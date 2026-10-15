# RetroLens — Performance Refresh

Versi ini difokuskan untuk frame pacing yang lebih stabil pada Android/Chrome.

## Menjalankan
1. Ekstrak folder proyek.
2. Jalankan melalui `localhost` (misalnya Live Server di Acode atau `python -m http.server 8080` di folder proyek).
3. Buka alamat localhost di Chrome dan izinkan akses kamera.

## Optimasi yang diterapkan
- Resolusi pemrosesan default 640×360 untuk menurunkan biaya pixel processing pada perangkat mobile.
- Inferensi hand landmarks dibatasi sekitar 30 Hz, sementara presentasi canvas tetap mengikuti `requestAnimationFrame`. Landmark terakhir dipakai di antara inferensi agar efek tidak berkedip.
- Canvas ROI portal dipakai ulang, bukan dibuat ulang setiap frame.
- Canvas context dioptimalkan untuk rendering kamera dan CSS canvas dipisahkan dari layout HUD.
- Versi MediaPipe dipatok ke `0.10.22` agar dependency tidak berubah diam-diam.
- Error saat satu frame deteksi dilewati tidak menghentikan render loop.

## Tentang target 60 FPS
Loop tampilan mengikuti `requestAnimationFrame` dan menargetkan 60 FPS pada layar 60 Hz. FPS nyata tetap bergantung pada GPU/CPU, browser, pencahayaan, suhu perangkat, dan filter yang aktif; JavaScript tidak bisa menjamin 60 FPS terkunci pada semua perangkat. Filter seperti edge, sketch, dan cartoon lebih berat daripada filter sederhana.

## Gestur
- Pinch ibu jari + telunjuk: pindah filter.
- Dua tangan mengepal: ganti mode 2D/3D.
- Tombol navigasi filter dan snapshot tetap tersedia.
