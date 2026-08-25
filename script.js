import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";


// ============================================================
// CONFIG
// ============================================================

const CFG = {
  frameWidth: 960,
  frameHeight: 540,

  pinchThresholdPx: 45,

  filterCooldownSec: 0.15,

  modeCooldownSec: 1.2,

  fistDistThresholdPx: 80,
};


// ============================================================
// FILTERS
// ============================================================

const FILTER_KEYS = [
  "dual-tone",
  "thermal",
  "sketch",
  "pixelate",
  "glitch",
  "invert",
  "red-channel",
  "edge",
  "blur",
  "cartoon",
  "rainbow-wave"
];


// ============================================================
// DOM
// ============================================================

const video =
  document.getElementById("video");

const outCanvas =
  document.getElementById("out");

const outCtx =
  outCanvas.getContext("2d", {
    willReadFrequently: true
  });

const placeholder =
  document.getElementById("placeholder");

const placeholderMsg =
  document.getElementById("placeholderMsg");

const handHintEl =
  document.getElementById("handHint");

const startBtn =
  document.getElementById("startBtn");

const stopBtn =
  document.getElementById("stopBtn");

const snapBtn =
  document.getElementById("snapBtn");

const prevBtn =
  document.getElementById("prevBtn");

const nextBtn =
  document.getElementById("nextBtn");

const modeBtn =
  document.getElementById("modeBtn");

const modeLabel =
  document.getElementById("modeLabel");

const filterIdxEl =
  document.getElementById("filterIdx");

const filterNameEl =
  document.getElementById("filterName");

const timecodeEl =
  document.getElementById("timecode");

const fistmeterEl =
  document.getElementById("fistmeter");

const filterStrip =
  document.getElementById("filterStrip");


outCanvas.width =
  CFG.frameWidth;

outCanvas.height =
  CFG.frameHeight;


// ============================================================
// DISPLAY NAME
// ============================================================

function displayName(key) {

  return key
    .split("-")
    .map(
      w =>
        w[0].toUpperCase() +
        w.slice(1)
    )
    .join(" ");
}


// ============================================================
// FILTER CHIP UI
// ============================================================

FILTER_KEYS.forEach((name, i) => {

  const chip =
    document.createElement("span");

  chip.className =
    "filter-chip" +
    (i === 0 ? " active" : "");

  chip.textContent =
    displayName(name);

  chip.onclick = () => {

    state.activeFilterIdx = i;

  };

  filterStrip.appendChild(chip);

});


// ============================================================
// STATE
// ============================================================

const state = {

  running: false,

  activeFilterIdx: 0,

  is3D: false,

  lastSwitch: 0,

  lastModeToggle: 0,

  startedAt: 0

};


let fistCountLast = 0;

let lastHandSeenAt = 0;

let lastFrameTime = 0;


// ============================================================
// FILTER STATE
// ============================================================

function currentFilterName() {

  return FILTER_KEYS[
    state.activeFilterIdx
  ];

}


function secondaryFilterName() {

  return FILTER_KEYS[
    (state.activeFilterIdx + 1) %
    FILTER_KEYS.length
  ];

}


function cycleFilter(step) {

  state.activeFilterIdx =
    (
      state.activeFilterIdx +
      step +
      FILTER_KEYS.length
    ) %
    FILTER_KEYS.length;

}


// ============================================================
// GEOMETRY
// ============================================================

function dist(a, b) {

  return Math.hypot(
    a.x - b.x,
    a.y - b.y
  );

}


// ============================================================
// FIST DETECTION
// ============================================================

function isFistClosed(
  landmarks,
  w,
  h,
  threshold
) {

  const wrist = {
    x: landmarks[0].x * w,
    y: landmarks[0].y * h
  };

  const tipIdx = [
    8,
    12,
    16,
    20
  ];

  let sum = 0;

  for (const t of tipIdx) {

    const p = {
      x: landmarks[t].x * w,
      y: landmarks[t].y * h
    };

    sum += dist(
      p,
      wrist
    );

  }

  return (
    sum / tipIdx.length
  ) < threshold;

}


// ============================================================
// PORTAL POINT SORTING
//
// FIX:
// Sebelumnya titik diurutkan berdasarkan X sehingga ketika
// dua tangan miring, polygon bisa menjadi bowtie / menyilang.
//
// Sekarang titik diurutkan berdasarkan sudut terhadap pusat,
// sehingga urutannya mengikuti perimeter polygon.
// ============================================================

function sortQuad(pts) {

  if (
    !pts ||
    pts.length !== 4
  ) {

    return pts;

  }


  // ----------------------------------------------------------
  // Cari pusat 4 titik
  // ----------------------------------------------------------

  const cx =
    pts.reduce(
      (sum, p) =>
        sum + p.x,
      0
    ) / 4;

  const cy =
    pts.reduce(
      (sum, p) =>
        sum + p.y,
      0
    ) / 4;


  // ----------------------------------------------------------
  // Urutkan berdasarkan sudut
  // ----------------------------------------------------------

  const sorted =
    [...pts].sort(
      (a, b) => {

        const angleA =
          Math.atan2(
            a.y - cy,
            a.x - cx
          );

        const angleB =
          Math.atan2(
            b.y - cy,
            b.x - cx
          );

        return angleA - angleB;

      }
    );


  // ----------------------------------------------------------
  // Cari titik paling atas,
  // lalu paling kiri jika hampir sama
  // ----------------------------------------------------------

  let start = 0;

  for (
    let i = 1;
    i < 4;
    i++
  ) {

    const a = sorted[i];

    const b = sorted[start];


    if (
      a.y < b.y ||
      (
        Math.abs(
          a.y - b.y
        ) < 10 &&
        a.x < b.x
      )
    ) {

      start = i;

    }

  }


  // ----------------------------------------------------------
  // Putar array agar dimulai dari kiri/atas
  // ----------------------------------------------------------

  return [

    sorted[start % 4],

    sorted[
      (start + 1) % 4
    ],

    sorted[
      (start + 2) % 4
    ],

    sorted[
      (start + 3) % 4
    ]

  ];

}


// ============================================================
// GRAYSCALE
// ============================================================

function grayscaleCanvas(src) {

  const g =
    document.createElement(
      "canvas"
    );

  g.width =
    src.width;

  g.height =
    src.height;


  const gc =
    g.getContext("2d");

  gc.filter =
    "grayscale(1)";

  gc.drawImage(
    src,
    0,
    0
  );

  return g;

}


// ============================================================
// JET LUT
// ============================================================

let JET_LUT = null;


function clamp(
  v,
  lo,
  hi
) {

  return v < lo
    ? lo
    : (
      v > hi
        ? hi
        : v
    );

}


function jetLUT() {

  if (JET_LUT) {

    return JET_LUT;

  }


  JET_LUT =
    new Uint8ClampedArray(
      256 * 3
    );


  for (
    let i = 0;
    i < 256;
    i++
  ) {

    const v =
      i / 255;


    const r =
      clamp(
        1.5 -
        Math.abs(
          4 * v - 3
        ),
        0,
        1
      );


    const g =
      clamp(
        1.5 -
        Math.abs(
          4 * v - 2
        ),
        0,
        1
      );


    const b =
      clamp(
        1.5 -
        Math.abs(
          4 * v - 1
        ),
        0,
        1
      );


    JET_LUT[i * 3] =
      r * 255;

    JET_LUT[i * 3 + 1] =
      g * 255;

    JET_LUT[i * 3 + 2] =
      b * 255;

  }


  return JET_LUT;

}


// ============================================================
// FILTER BANK
// ============================================================

const FilterBank = {

  // ----------------------------------------------------------
  // DUAL TONE
  // ----------------------------------------------------------

  "dual-tone"(src) {

    const w =
      src.width;

    const h =
      src.height;


    const gray =
      grayscaleCanvas(src);


    const gd =
      gray
        .getContext("2d")
        .getImageData(
          0,
          0,
          w,
          h
        );


    const out =
      document.createElement(
        "canvas"
      );

    out.width = w;
    out.height = h;


    const octx =
      out.getContext("2d");


    const od =
      octx.createImageData(
        w,
        h
      );


    for (
      let i = 0;
      i < gd.data.length;
      i += 4
    ) {

      const v =
        gd.data[i];


      if (v > 110) {

        od.data[i] =
          255;

        od.data[i + 1] =
          140;

        od.data[i + 2] =
          10;

      } else {

        od.data[i] =
          220;

        od.data[i + 1] =
          30;

        od.data[i + 2] =
          180;

      }

      od.data[i + 3] =
        255;

    }


    octx.putImageData(
      od,
      0,
      0
    );


    return out;

  },


  // ----------------------------------------------------------
  // THERMAL
  // ----------------------------------------------------------

  "thermal"(src) {

    const w =
      src.width;

    const h =
      src.height;


    const gray =
      grayscaleCanvas(src);


    const gd =
      gray
        .getContext("2d")
        .getImageData(
          0,
          0,
          w,
          h
        );


    const lut =
      jetLUT();


    const out =
      document.createElement(
        "canvas"
      );

    out.width = w;
    out.height = h;


    const octx =
      out.getContext("2d");


    const od =
      octx.createImageData(
        w,
        h
      );


    for (
      let i = 0;
      i < gd.data.length;
      i += 4
    ) {

      const v =
        gd.data[i];


      od.data[i] =
        lut[v * 3];

      od.data[i + 1] =
        lut[v * 3 + 1];

      od.data[i + 2] =
        lut[v * 3 + 2];

      od.data[i + 3] =
        255;

    }


    octx.putImageData(
      od,
      0,
      0
    );


    return out;

  },


  // ----------------------------------------------------------
  // SKETCH
  // ----------------------------------------------------------

  "sketch"(src) {

    const w =
      src.width;

    const h =
      src.height;


    const gray =
      grayscaleCanvas(src);


    const gd =
      gray
        .getContext("2d")
        .getImageData(
          0,
          0,
          w,
          h
        );


    const blurCv =
      document.createElement(
        "canvas"
      );

    blurCv.width = w;
    blurCv.height = h;


    const bctx =
      blurCv.getContext("2d");


    bctx.filter =
      "invert(1) blur(6px)";


    bctx.drawImage(
      gray,
      0,
      0
    );


    const bd =
      bctx.getImageData(
        0,
        0,
        w,
        h
      );


    const out =
      document.createElement(
        "canvas"
      );

    out.width = w;
    out.height = h;


    const octx =
      out.getContext("2d");


    const od =
      octx.createImageData(
        w,
        h
      );


    for (
      let i = 0;
      i < gd.data.length;
      i += 4
    ) {

      const g =
        gd.data[i];

      const bInv =
        bd.data[i];


      const val =
        clamp(
          (
            g * 255
          ) /
          (
            255 -
            bInv +
            1
          ),
          0,
          255
        );


      od.data[i] =
        val;

      od.data[i + 1] =
        val;

      od.data[i + 2] =
        val;

      od.data[i + 3] =
        255;

    }


    octx.putImageData(
      od,
      0,
      0
    );


    return out;

  },


  // ----------------------------------------------------------
  // PIXELATE
  // ----------------------------------------------------------

  "pixelate"(src, block = 14) {

    const w =
      src.width;

    const h =
      src.height;


    if (
      w < 2 ||
      h < 2
    ) {

      return src;

    }


    const sw =
      Math.max(
        1,
        Math.floor(
          w / block
        )
      );


    const sh =
      Math.max(
        1,
        Math.floor(
          h / block
        )
      );


    const small =
      document.createElement(
        "canvas"
      );


    small.width = sw;
    small.height = sh;


    small
      .getContext("2d")
      .drawImage(
        src,
        0,
        0,
        w,
        h,
        0,
        0,
        sw,
        sh
      );


    const out =
      document.createElement(
        "canvas"
      );


    out.width = w;
    out.height = h;


    const octx =
      out.getContext("2d");


    octx.imageSmoothingEnabled =
      false;


    octx.drawImage(
      small,
      0,
      0,
      sw,
      sh,
      0,
      0,
      w,
      h
    );


    return out;

  },


  // ----------------------------------------------------------
  // GLITCH
  // ----------------------------------------------------------

  "glitch"(src) {

    const w =
      src.width;

    const h =
      src.height;


    if (
      w < 2 ||
      h < 2
    ) {

      return src;

    }


    const sctx =
      src.getContext("2d");


    const sd =
      sctx.getImageData(
        0,
        0,
        w,
        h
      );


    const out =
      document.createElement(
        "canvas"
      );

    out.width = w;
    out.height = h;


    const octx =
      out.getContext("2d");


    const od =
      octx.createImageData(
        w,
        h
      );


    const shift =
      4 +
      Math.floor(
        Math.random() * 9
      );


    for (
      let y = 0;
      y < h;
      y++
    ) {

      for (
        let x = 0;
        x < w;
        x++
      ) {

        const di =
          (y * w + x) * 4;


        const rx =
          (
            x -
            shift +
            w
          ) % w;


        const bx =
          (
            x +
            shift
          ) % w;


        const ri =
          (
            y * w + rx
          ) * 4;


        const bi =
          (
            y * w + bx
          ) * 4;


        od.data[di] =
          sd.data[bi];

        od.data[di + 1] =
          sd.data[di + 1];

        od.data[di + 2] =
          sd.data[ri + 2];

        od.data[di + 3] =
          255;

      }

    }


    octx.putImageData(
      od,
      0,
      0
    );


    for (
      let n = 0;
      n < 2;
      n++
    ) {

      const y =
        Math.floor(
          Math.random() * h
        );


      const noise =
        octx.createImageData(
          w,
          1
        );


      for (
        let i = 0;
        i < noise.data.length;
        i += 4
      ) {

        noise.data[i] =
          Math.random() * 255;

        noise.data[i + 1] =
          Math.random() * 255;

        noise.data[i + 2] =
          Math.random() * 255;

        noise.data[i + 3] =
          255;

      }


      octx.putImageData(
        noise,
        0,
        y
      );

    }


    return out;

  },


  // ----------------------------------------------------------
  // INVERT
  // ----------------------------------------------------------

  "invert"(src) {

    const out =
      document.createElement(
        "canvas"
      );


    out.width =
      src.width;

    out.height =
      src.height;


    const octx =
      out.getContext("2d");


    octx.filter =
      "invert(1)";


    octx.drawImage(
      src,
      0,
      0
    );


    return out;

  },


  // ----------------------------------------------------------
  // RED CHANNEL
  // ----------------------------------------------------------

  "red-channel"(src) {

    const w =
      src.width;

    const h =
      src.height;


    const sd =
      src
        .getContext("2d")
        .getImageData(
          0,
          0,
          w,
          h
        );


    const out =
      document.createElement(
        "canvas"
      );


    out.width = w;
    out.height = h;


    const octx =
      out.getContext("2d");


    const od =
      octx.createImageData(
        w,
        h
      );


    for (
      let i = 0;
      i < sd.data.length;
      i += 4
    ) {

      od.data[i] =
        sd.data[i];

      od.data[i + 1] =
        0;

      od.data[i + 2] =
        0;

      od.data[i + 3] =
        255;

    }


    octx.putImageData(
      od,
      0,
      0
    );


    return out;

  },


  // ----------------------------------------------------------
  // EDGE
  // ----------------------------------------------------------

  "edge"(src) {

    const w =
      src.width;

    const h =
      src.height;


    const gray =
      grayscaleCanvas(src);


    const gd =
      gray
        .getContext("2d")
        .getImageData(
          0,
          0,
          w,
          h
        );


    const g =
      gd.data;


    const lut =
      jetLUT();


    const out =
      document.createElement(
        "canvas"
      );


    out.width = w;
    out.height = h;


    const octx =
      out.getContext("2d");


    const od =
      octx.createImageData(
        w,
        h
      );


    const at =
      (x, y) =>
        g[
          (
            clamp(
              y,
              0,
              h - 1
            ) *
            w +
            clamp(
              x,
              0,
              w - 1
            )
          ) * 4
        ];


    for (
      let y = 0;
      y < h;
      y++
    ) {

      for (
        let x = 0;
        x < w;
        x++
      ) {

        const gx =
          -at(x - 1, y - 1)
          - 2 * at(x - 1, y)
          - at(x - 1, y + 1)
          + at(x + 1, y - 1)
          + 2 * at(x + 1, y)
          + at(x + 1, y + 1);


        const gy =
          -at(x - 1, y - 1)
          - 2 * at(x, y - 1)
          - at(x + 1, y - 1)
          + at(x - 1, y + 1)
          + 2 * at(x, y + 1)
          + at(x + 1, y + 1);


        const mag =
          Math.min(
            255,
            Math.sqrt(
              gx * gx +
              gy * gy
            )
          );


        const i =
          (y * w + x) * 4;


        if (mag > 70) {

          od.data[i] =
            lut[
              Math.floor(mag) * 3
            ] * 0.3;

          od.data[i + 1] =
            200;

          od.data[i + 2] =
            40;

          od.data[i + 3] =
            255;

        } else {

          od.data[i] =
            0;

          od.data[i + 1] =
            0;

          od.data[i + 2] =
            0;

          od.data[i + 3] =
            255;

        }

      }

    }


    octx.putImageData(
      od,
      0,
      0
    );


    return out;

  },


  // ----------------------------------------------------------
  // BLUR
  // ----------------------------------------------------------

  "blur"(src) {

    const out =
      document.createElement(
        "canvas"
      );


    out.width =
      src.width;

    out.height =
      src.height;


    const octx =
      out.getContext("2d");


    octx.filter =
      "blur(9px)";


    octx.drawImage(
      src,
      0,
      0
    );


    return out;

  },


  // ----------------------------------------------------------
  // CARTOON
  // ----------------------------------------------------------

  "cartoon"(src) {

    const w =
      src.width;

    const h =
      src.height;


    const color =
      document.createElement(
        "canvas"
      );


    color.width = w;
    color.height = h;


    const cctx =
      color.getContext("2d");


    cctx.filter =
      "saturate(1.5) contrast(1.08) blur(1.5px)";


    cctx.drawImage(
      src,
      0,
      0
    );


    const edges =
      FilterBank["edge"](src);


    const ed =
      edges
        .getContext("2d")
        .getImageData(
          0,
          0,
          w,
          h
        );


    const cd =
      cctx.getImageData(
        0,
        0,
        w,
        h
      );


    const od =
      cctx.createImageData(
        w,
        h
      );


    for (
      let i = 0;
      i < cd.data.length;
      i += 4
    ) {

      const isEdge =
        ed.data[i + 1] >
        40;


      const f =
        isEdge
          ? 0.35
          : 1.0;


      od.data[i] =
        cd.data[i] * f;

      od.data[i + 1] =
        cd.data[i + 1] * f;

      od.data[i + 2] =
        cd.data[i + 2] * f;

      od.data[i + 3] =
        255;

    }


    cctx.putImageData(
      od,
      0,
      0
    );


    return color;

  },


  // ----------------------------------------------------------
  // RAINBOW WAVE
  // ----------------------------------------------------------

  "rainbow-wave"(src) {

    const w =
      src.width;

    const h =
      src.height;


    const sd =
      src
        .getContext("2d")
        .getImageData(
          0,
          0,
          w,
          h
        );


    const lut =
      jetLUT();


    const t =
      performance.now() /
      1000 *
      5.0;


    const out =
      document.createElement(
        "canvas"
      );


    out.width = w;
    out.height = h;


    const octx =
      out.getContext("2d");


    const od =
      octx.createImageData(
        w,
        h
      );


    for (
      let y = 0;
      y < h;
      y++
    ) {

      for (
        let x = 0;
        x < w;
        x++
      ) {

        const i =
          (y * w + x) * 4;


        const pattern =
          Math.sin(
            (x + y) *
            0.05 +
            t
          ) *
          127 +
          128;


        const idx =
          clamp(
            Math.floor(pattern),
            0,
            255
          ) * 3;


        od.data[i] =
          sd.data[i] * 0.3 +
          lut[idx] * 0.7;


        od.data[i + 1] =
          sd.data[i + 1] * 0.3 +
          lut[idx + 1] * 0.7;


        od.data[i + 2] =
          sd.data[i + 2] * 0.3 +
          lut[idx + 2] * 0.7;


        od.data[i + 3] =
          255;

      }

    }


    octx.putImageData(
      od,
      0,
      0
    );


    return out;

  }

};


// ============================================================
// PORTAL RENDER
// ============================================================

function renderPortal(
  baseCanvas,
  pts,
  filterKey
) {

  if (
    !pts ||
    pts.length < 4
  ) {

    return;

  }


  const xs =
    pts.map(
      p => p.x
    );

  const ys =
    pts.map(
      p => p.y
    );


  let x =
    Math.max(
      0,
      Math.min(...xs)
    );


  let y =
    Math.max(
      0,
      Math.min(...ys)
    );


  let x2 =
    Math.min(
      baseCanvas.width,
      Math.max(...xs)
    );


  let y2 =
    Math.min(
      baseCanvas.height,
      Math.max(...ys)
    );


  const w =
    x2 - x;

  const h =
    y2 - y;


  if (
    w <= 10 ||
    h <= 10
  ) {

    return;

  }


  const roi =
    document.createElement(
      "canvas"
    );


  roi.width =
    Math.max(
      1,
      Math.floor(w)
    );

  roi.height =
    Math.max(
      1,
      Math.floor(h)
    );


  roi
    .getContext("2d")
    .drawImage(
      baseCanvas,
      x,
      y,
      w,
      h,
      0,
      0,
      roi.width,
      roi.height
    );


  let processed;


  try {

    processed =
      FilterBank[
        filterKey
      ](roi);

  } catch (e) {

    console.warn(
      "Filter error:",
      e
    );

    processed =
      roi;

  }


  // ----------------------------------------------------------
  // CLIP PORTAL
  // ----------------------------------------------------------

  outCtx.save();


  outCtx.beginPath();


  outCtx.moveTo(
    pts[0].x,
    pts[0].y
  );


  for (
    let i = 1;
    i < pts.length;
    i++
  ) {

    outCtx.lineTo(
      pts[i].x,
      pts[i].y
    );

  }


  outCtx.closePath();


  outCtx.clip();


  outCtx.drawImage(
    processed,
    x,
    y,
    w,
    h
  );


  outCtx.restore();


  // ----------------------------------------------------------
  // PORTAL BORDER
  // ----------------------------------------------------------

  outCtx.save();


  outCtx.beginPath();


  outCtx.moveTo(
    pts[0].x,
    pts[0].y
  );


  for (
    let i = 1;
    i < pts.length;
    i++
  ) {

    outCtx.lineTo(
      pts[i].x,
      pts[i].y
    );

  }


  outCtx.closePath();


  outCtx.strokeStyle =
    "#ffffff";

  outCtx.lineWidth =
    2;


  outCtx.stroke();


  outCtx.restore();

}


// ============================================================
// HAND SKELETON
// ============================================================

const HAND_CONNECTIONS = [

  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],

  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],

  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],

  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],

  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],

  [0, 17]

];


function drawSkeleton(px) {

  outCtx.strokeStyle =
    "rgba(57,255,136,0.85)";

  outCtx.lineWidth =
    1.4;


  outCtx.beginPath();


  for (
    const [a, b]
    of HAND_CONNECTIONS
  ) {

    outCtx.moveTo(
      px[a].x,
      px[a].y
    );

    outCtx.lineTo(
      px[b].x,
      px[b].y
    );

  }


  outCtx.stroke();


  outCtx.fillStyle =
    "#ffb32c";


  for (
    const p of px
  ) {

    outCtx.beginPath();


    outCtx.arc(
      p.x,
      p.y,
      2,
      0,
      Math.PI * 2
    );


    outCtx.fill();

  }

}


// ============================================================
// MIRROR LANDMARKS
// ============================================================

function toMirroredPx(
  lm,
  W,
  H
) {

  return lm.map(
    p => ({

      x:
        (1 - p.x) *
        W,

      y:
        p.y *
        H

    })
  );

}


// ============================================================
// MEDIAPIPE
// ============================================================

let handLandmarker =
  null;


let lastVideoTime =
  -1;


async function loadHandLandmarker() {

  placeholderMsg.textContent =
    "Memuat model deteksi tangan...";


  const vision =
    await FilesetResolver
      .forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );


  const baseOpts = {

    baseOptions: {

      modelAssetPath:
        "models/hand_landmarker.task",

      delegate:
        "GPU"

    },

    runningMode:
      "VIDEO",

    numHands:
      2,

    minHandDetectionConfidence:
      0.5,

    minHandPresenceConfidence:
      0.5,

    minTrackingConfidence:
      0.5

  };


  try {

    handLandmarker =
      await HandLandmarker
        .createFromOptions(
          vision,
          baseOpts
        );

  } catch (e) {

    console.warn(
      "GPU delegate gagal, coba CPU...",
      e
    );


    baseOpts
      .baseOptions
      .delegate =
      "CPU";


    handLandmarker =
      await HandLandmarker
        .createFromOptions(
          vision,
          baseOpts
        );

  }

}


// ============================================================
// PROCESS FRAME
// ============================================================

function processFrame() {

  const W =
    CFG.frameWidth;

  const H =
    CFG.frameHeight;


  const now =
    performance.now() /
    1000;


  // ----------------------------------------------------------
  // CLEAR
  // ----------------------------------------------------------

  outCtx.clearRect(
    0,
    0,
    W,
    H
  );


  // ----------------------------------------------------------
  // MIRROR CAMERA
  // ----------------------------------------------------------

  outCtx.save();


  outCtx.translate(
    W,
    0
  );


  outCtx.scale(
    -1,
    1
  );


  outCtx.drawImage(
    video,
    0,
    0,
    W,
    H
  );


  outCtx.restore();


  // ----------------------------------------------------------
  // DETECTION
  // ----------------------------------------------------------

  let results = null;


  if (
    handLandmarker &&
    video.currentTime !==
      lastVideoTime
  ) {

    lastVideoTime =
      video.currentTime;


    results =
      handLandmarker
        .detectForVideo(
          video,
          performance.now()
        );

  }


  const allHandTips = [];


  let fistCount =
    0;


  if (
    results &&
    results.landmarks &&
    results.landmarks.length
  ) {

    lastHandSeenAt =
      now;


    // --------------------------------------------------------
    // EACH HAND
    // --------------------------------------------------------

    for (
      const lm of results.landmarks
    ) {

      const px =
        toMirroredPx(
          lm,
          W,
          H
        );


      // Skeleton
      drawSkeleton(px);


      // Tips:
      // 4  = thumb
      // 8  = index
      // 12 = middle
      // 16 = ring
      // 20 = pinky

      const tips = [

        px[4],
        px[8],
        px[12],
        px[16],
        px[20]

      ];


      allHandTips.push(
        tips
      );


      // ------------------------------------------------------
      // PINCH
      // ------------------------------------------------------

      if (
        dist(
          tips[0],
          tips[1]
        ) <
        CFG.pinchThresholdPx
      ) {

        if (
          now -
          state.lastSwitch >
          CFG.filterCooldownSec
        ) {

          cycleFilter(1);

          state.lastSwitch =
            now;

        }

      }


      // ------------------------------------------------------
      // FIST
      // ------------------------------------------------------

      if (
        isFistClosed(
          px,
          1,
          1,
          CFG.fistDistThresholdPx
        )
      ) {

        fistCount++;

      }

    }


    // --------------------------------------------------------
    // MODE TOGGLE
    // --------------------------------------------------------

    if (
      fistCount === 2 &&
      now -
        state.lastModeToggle >
        CFG.modeCooldownSec
    ) {

      state.is3D =
        !state.is3D;


      state.lastModeToggle =
        now;

    }


    // ========================================================
    // 3D MODE
    // ========================================================

    if (state.is3D) {

      if (
        allHandTips.length === 2
      ) {

        const [
          t1,
          t2
        ] =
          allHandTips;


        renderPortal(

          outCanvas,

          [
            t1[0],
            t1[1],
            t1[2],

            t2[2],
            t2[1],
            t2[0]
          ],

          currentFilterName()

        );


        renderPortal(

          outCanvas,

          [
            t1[2],
            t1[3],
            t1[4],

            t2[4],
            t2[3],
            t2[2]
          ],

          secondaryFilterName()

        );


      } else if (
        allHandTips.length === 1
      ) {

        renderPortal(

          outCanvas,

          [
            allHandTips[0][0],
            allHandTips[0][1],
            allHandTips[0][2],
            allHandTips[0][4]
          ],

          currentFilterName()

        );

      }

    }


    // ========================================================
    // 2D MODE
    // ========================================================

    else {

      // ------------------------------------------------------
      // TWO HANDS
      // ------------------------------------------------------

      if (
        allHandTips.length === 2
      ) {

        const corners = [

          allHandTips[0][0],
          allHandTips[0][1],

          allHandTips[1][0],
          allHandTips[1][1]

        ];


        // ====================================================
        // FIX CROSSING PORTAL
        //
        // Jangan lagi pakai sort berdasarkan X.
        // sortQuad() mengurutkan titik berdasarkan perimeter.
        // ====================================================

        const quad =
          sortQuad(
            corners
          );


        renderPortal(

          outCanvas,

          quad,

          currentFilterName()

        );

      }


      // ------------------------------------------------------
      // ONE HAND
      // ------------------------------------------------------

      else if (
        allHandTips.length === 1
      ) {

        const t =
          allHandTips[0];


        const quad =
          sortQuad([

            t[0],
            t[1],
            t[2],
            t[4]

          ]);


        renderPortal(

          outCanvas,

          quad,

          currentFilterName()

        );

      }

    }

  }


  fistCountLast =
    fistCount;


  updateHUD(now);

}


// ============================================================
// HUD
// ============================================================

function updateHUD(now) {

  modeLabel.textContent =
    state.is3D
      ? "3D"
      : "2D";


  modeBtn.textContent =
    state.is3D
      ? "Ganti ke 2D"
      : "Ganti ke 3D";


  filterIdxEl.textContent =
    (
      state.activeFilterIdx +
      1
    ) +
    "/" +
    FILTER_KEYS.length;


  filterNameEl.textContent =
    displayName(
      currentFilterName()
    );


  // Active filter

  [
    ...filterStrip.children
  ].forEach(
    (chip, i) => {

      chip.classList.toggle(
        "active",
        i ===
          state.activeFilterIdx
      );

    }
  );


  // Fist indicator

  [
    ...fistmeterEl.children
  ].forEach(
    (el, i) => {

      el.classList.toggle(
        "on",
        i < fistCountLast
      );

    }
  );


  // Timer

  if (
    state.startedAt
  ) {

    const secs =
      Math.floor(
        performance.now() /
          1000 -
        state.startedAt
      );


    const mm =
      String(
        Math.floor(
          secs / 60
        )
      ).padStart(
        2,
        "0"
      );


    const ss =
      String(
        secs % 60
      ).padStart(
        2,
        "0"
      );


    timecodeEl.textContent =
      `${mm}:${ss}`;

  }


  // Hand hint

  if (
    state.running &&
    now !== undefined &&
    state.startedAt &&
    now -
      state.startedAt >
      4
  ) {

    handHintEl.style.display =
      (
        now -
        lastHandSeenAt >
        3
      )
        ? "block"
        : "none";

  }

}


// ============================================================
// CAMERA LOOP
// ============================================================

let rafId =
  null;


function loop() {

  if (
    state.running &&
    video.readyState >= 2 &&
    handLandmarker
  ) {

    processFrame();

  }


  rafId =
    requestAnimationFrame(
      loop
    );

}


// ============================================================
// ERROR
// ============================================================

function showPlaceholderError(
  html
) {

  placeholder.style.display =
    "flex";


  placeholderMsg.innerHTML =
    html;

}


// ============================================================
// START CAMERA
// ============================================================

async function startCamera() {

  try {

    const stream =
      await navigator
        .mediaDevices
        .getUserMedia({

          video: {

            width: {
              ideal:
                CFG.frameWidth
            },

            height: {
              ideal:
                CFG.frameHeight
            },

            facingMode:
              "user"

          },

          audio:
            false

        });


    video.srcObject =
      stream;


    await video.play();


    // --------------------------------------------------------
    // LOAD MODEL
    // --------------------------------------------------------

    if (
      !handLandmarker
    ) {

      try {

        await loadHandLandmarker();

      } catch (e) {

        console.error(
          "Gagal memuat HandLandmarker:",
          e
        );


        showPlaceholderError(

          `Gagal memuat model tangan.<br>
          <span style="color:#ff5c5c">
          ${
            (
              e &&
              e.message
            ) || e
          }
          </span><br>
          Pastikan file
          <code>
          models/hand_landmarker.task
          </code>
          ada di folder proyek dan halaman dibuka lewat server lokal.`

        );


        stream
          .getTracks()
          .forEach(
            t =>
              t.stop()
          );


        return;

      }

    }


    // --------------------------------------------------------
    // RUNNING
    // --------------------------------------------------------

    state.running =
      true;


    state.startedAt =
      performance.now() /
      1000;


    lastHandSeenAt =
      0;


    placeholder.style.display =
      "none";


    startBtn.disabled =
      true;


    stopBtn.disabled =
      false;


    snapBtn.disabled =
      false;


    prevBtn.disabled =
      false;


    nextBtn.disabled =
      false;


    modeBtn.disabled =
      false;


    loop();

  } catch (err) {

    console.error(
      "getUserMedia failed:",
      err
    );


    showPlaceholderError(

      `Tidak bisa mengakses kamera.<br>
      <span style="color:#ff5c5c">
      ${
        err.message ||
        err
      }
      </span><br>
      Pastikan halaman dibuka lewat HTTPS atau localhost,
      dan izin kamera diberikan.`

    );

  }

}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {

  state.running =
    false;


  if (rafId) {

    cancelAnimationFrame(
      rafId
    );

    rafId =
      null;

  }


  const stream =
    video.srcObject;


  if (stream) {

    stream
      .getTracks()
      .forEach(
        t =>
          t.stop()
      );

  }


  video.srcObject =
    null;


  outCtx.clearRect(
    0,
    0,
    CFG.frameWidth,
    CFG.frameHeight
  );


  handHintEl.style.display =
    "none";


  placeholder.style.display =
    "flex";


  startBtn.disabled =
    false;


  stopBtn.disabled =
    true;


  snapBtn.disabled =
    true;


  prevBtn.disabled =
    true;


  nextBtn.disabled =
    true;


  modeBtn.disabled =
    true;

}


// ============================================================
// SNAPSHOT
// ============================================================

function snapshot() {

  const link =
    document.createElement(
      "a"
    );


  link.download =
    `retrolens_${Date.now()}.png`;


  link.href =
    outCanvas.toDataURL(
      "image/png"
    );


  link.click();

}


// ============================================================
// CONTROLS
// ============================================================

startBtn.onclick =
  startCamera;


stopBtn.onclick =
  stopCamera;


snapBtn.onclick =
  snapshot;


prevBtn.onclick =
  () =>
    cycleFilter(-1);


nextBtn.onclick =
  () =>
    cycleFilter(1);


modeBtn.onclick =
  () => {

    state.is3D =
      !state.is3D;

  };


// ============================================================
// KEYBOARD
// ============================================================

window.addEventListener(
  "keydown",
  e => {

    if (
      !state.running
    ) {

      return;

    }


    if (
      e.key === "c" ||
      e.key === "C"
    ) {

      state.is3D =
        !state.is3D;

    }


    else if (
      e.key === "n" ||
      e.key === "N"
    ) {

      cycleFilter(1);

    }


    else if (
      e.key === "p" ||
      e.key === "P"
    ) {

      cycleFilter(-1);

    }


    else if (
      e.key === "s" ||
      e.key === "S"
    ) {

      snapshot();

    }

  }
);


// ============================================================
// INITIAL HUD
// ============================================================

updateHUD();