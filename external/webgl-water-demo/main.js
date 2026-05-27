/*
 * WebGL Water
 * https://madebyevan.com/webgl-water/
 *
 * Copyright 2011 Evan Wallace
 * Released under the MIT license
 */

function text2html(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
}

function handleError(text) {
  var html = text2html(text);
  if (html == 'WebGL not supported') {
    html = 'Your browser does not support WebGL.<br>Please see\
    <a href="https://get.webgl.org/get-a-webgl-implementation/">\
    Getting a WebGL Implementation</a>.';
  }
  var loading = document.getElementById('loading');
  loading.innerHTML = html;
  loading.style.zIndex = 1;
}

window.onerror = handleError;

var gl = GL.create();
var water;
var cubemap;
var renderer;
var angleX = -25;
var angleY = -200.5;

// Sphere physics info
var useSpherePhysics = false;
var center;
var oldCenter;
var velocity;
var gravity;
var radius;
var paused = false;

window.onload = function() {
  var ratio = window.devicePixelRatio || 1;
  var controls = document.getElementById('controls');
  var causticView = document.getElementById('causticView');
  var causticPreviewCanvas = document.getElementById('causticPreview');
  var causticPreviewCtx = causticPreviewCanvas ? causticPreviewCanvas.getContext('2d') : null;
  var causticReadFb = null;       /* reusable framebuffer for texture readback */
  var causticReadBuffer = null;   /* reusable Uint8Array buffer */
  var causticObservedMax = 0;     /* monotonically-growing max across frames; reset on .bin load */

  function onresize() {
    /* Canvas fills the window; collapsible panels float over it. */
    var width = innerWidth;
    var height = innerHeight;
    gl.canvas.width = width * ratio;
    gl.canvas.height = height * ratio;
    gl.canvas.style.width = width + 'px';
    gl.canvas.style.height = height + 'px';
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.matrixMode(gl.PROJECTION);
    gl.loadIdentity();
    gl.perspective(45, gl.canvas.width / gl.canvas.height, 0.01, 100);
    gl.matrixMode(gl.MODELVIEW);
    draw();
  }

  document.body.appendChild(gl.canvas);
  gl.clearColor(0, 0, 0, 1);

  water = new Water();
  renderer = new Renderer();

  /* Live pool-depth slider: value is in TANK-WIDTHS (1.0 = depth equals
   * lateral extent). Wallace's pool spans 2 world units laterally, so
   * we scale by 2.0 when setting the shader uniform. Re-runs the
   * caustic shader on every change because poolHeight enters the
   * caustic projection. */
  var slider = document.getElementById('poolHeight');
  var sliderVal = document.getElementById('poolHeightVal');
  if (slider && sliderVal) {
    var sync = function() {
      var h = parseFloat(slider.value);     /* tank-widths */
      renderer.poolHeight = h * 2.0;        /* convert to Wallace world units */
      sliderVal.textContent = h.toFixed(2);
      /* Re-bake the caustic texture against the new depth (only after
       * a .bin has been loaded; before that, calling updateCaustics
       * resizes the shared depth renderbuffer to 1024×1024, which then
       * trips Wallace's canDrawTo check on the 256×256 water textures). */
      if (water && water.anim) {
        renderer.updateCaustics(water);
        if (paused) draw();
      }
    };
    slider.addEventListener('input', sync);
    sync();
  }
  cubemap = new Cubemap({
    xneg: document.getElementById('xneg'),
    xpos: document.getElementById('xpos'),
    yneg: document.getElementById('ypos'),
    ypos: document.getElementById('ypos'),
    zneg: document.getElementById('zneg'),
    zpos: document.getElementById('zpos')
  });

  /* Top-down caustic preview: read renderer.causticTex back via
   * gl.readPixels, downsample to the 2D preview canvas. Static view —
   * doesn't rotate with the main scene. Called after every
   * updateCaustics() so the preview stays in sync. */
  function updateCausticPreview() {
    if (!causticPreviewCanvas || !renderer || !renderer.causticTex) return;
    if (!causticView || !causticView.open) return;   /* skip when collapsed */
    var srcW = renderer.causticTex.width;
    var srcH = renderer.causticTex.height;
    if (!causticReadFb) causticReadFb = gl.createFramebuffer();
    if (!causticReadBuffer) causticReadBuffer = new Uint8Array(srcW * srcH * 4);

    var prevFb = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.FRAMEBUFFER, causticReadFb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                             gl.TEXTURE_2D, renderer.causticTex.id, 0);
    gl.readPixels(0, 0, srcW, srcH, gl.RGBA, gl.UNSIGNED_BYTE, causticReadBuffer);
    gl.bindFramebuffer(gl.FRAMEBUFFER, prevFb);

    /* Wallace's caustic texture is laid out with the active pool floor
     * occupying the central 75% of the texture (the 0.75 factor in the
     * caustic vertex shader's gl_Position). Crop to that region so the
     * preview shows just the pool floor. */
    var crop = 0.125;
    var srcX0 = Math.floor(srcW * crop);
    var srcY0 = Math.floor(srcH * crop);
    var srcWUsed = srcW - 2 * srcX0;
    var srcHUsed = srcH - 2 * srcY0;

    var w = causticPreviewCanvas.width;
    var h = causticPreviewCanvas.height;
    var img = causticPreviewCtx.createImageData(w, h);

    /* Auto-scale via a monotonically-growing observed max across all
     * frames seen since the last .bin load. A per-frame max would
     * pulse on time-varying animations whenever the caustic peak
     * brightness changed; the growing-max approach converges to the
     * global max after the first cycle and stays stable thereafter.
     * Single pass over the cropped preview region. */
    var frameMax = 1;
    for (var y = 0; y < h; y++) {
      var sy = srcY0 + Math.floor((h - 1 - y) / h * srcHUsed);
      for (var x = 0; x < w; x++) {
        var sx = srcX0 + Math.floor(x / w * srcWUsed);
        var v = causticReadBuffer[(sy * srcW + sx) * 4];
        if (v > frameMax) frameMax = v;
      }
    }
    if (frameMax > causticObservedMax) causticObservedMax = frameMax;
    /* Map observed max to ~240 (headroom so peaks aren't pure-saturated). */
    var scale = 240.0 / causticObservedMax;

    for (var y2 = 0; y2 < h; y2++) {
      /* Flip y because WebGL's framebuffer y-axis is bottom-up */
      var sy2 = srcY0 + Math.floor((h - 1 - y2) / h * srcHUsed);
      for (var x2 = 0; x2 < w; x2++) {
        var sx2 = srcX0 + Math.floor(x2 / w * srcWUsed);
        var val = causticReadBuffer[(sy2 * srcW + sx2) * 4] * scale;
        if (val > 255) val = 255;
        var i = (y2 * w + x2) * 4;
        img.data[i]     = val;
        img.data[i + 1] = val;
        img.data[i + 2] = val;
        img.data[i + 3] = 255;
      }
    }
    causticPreviewCtx.putImageData(img, 0, 0);
  }

  /* Refresh preview when the user opens the panel (it might be stale) */
  if (causticView) {
    causticView.addEventListener('toggle', function() {
      if (causticView.open) updateCausticPreview();
    });
  }

  if (!water.textureA.canDrawTo() || !water.textureB.canDrawTo()) {
    throw new Error('Rendering to floating-point textures is required but not supported');
  }

  /* Sphere removed for Water_v2 caustic visualizer — hide by zero radius */
  center = oldCenter = new GL.Vector(-0.4, -0.75, 0.2);
  velocity = new GL.Vector();
  gravity = new GL.Vector(0, -4, 0);
  radius = 0;  /* hide sphere */

  /* Animation playback state */
  var animTime = 0;
  var animLastWallClock = 0;
  var heightScale = 1.0;

  /* Animation file picker */
  var fileInput = document.getElementById('animFile');
  if (fileInput) {
    fileInput.addEventListener('change', function(e) {
      var f = e.target.files[0];
      if (!f) return;
      var reader = new FileReader();
      reader.onload = function(ev) {
        try {
          var info = water.loadAnimation(ev.target.result);
          var label = document.getElementById('animLabel');
          if (label) {
            label.textContent = f.name + ' (' + info.nFrames + ' frames, ' +
                                info.nx + 'x' + info.ny + ')';
          }
          animTime = 0;
          /* Reset preview's observed-max so brightness normalization
           * recalibrates to this new caustic, not the previous one. */
          causticObservedMax = 0;
          /* Reset height scale to 1.0 so the loaded frame plays at native
           * amplitude (heightScale = 1.0 means raw values from the .bin
           * are uploaded unchanged). */
          heightScale = 1.0;
          var hsSlider = document.getElementById('heightScale');
          var hsVal = document.getElementById('heightScaleVal');
          if (hsSlider) hsSlider.value = heightScale;
          if (hsVal) hsVal.textContent = heightScale.toFixed(2);
          /* Reset time controls */
          var ts = document.getElementById('timeScrub');
          var tl = document.getElementById('timeLabel');
          if (ts) ts.value = 0;
          if (tl) {
            var p = info.period_s > 0 ? info.period_s : 1.0;
            if (info.nFrames > 1) {
              tl.textContent = 't = 0.00 / ' + p.toFixed(2) + ' s  (' +
                               info.nFrames + ' frames)';
            } else {
              tl.textContent = '(static — single frame)';
            }
          }
          water.playFrame(0, heightScale);
          renderer.updateCaustics(water);
          draw();
        } catch (err) {
          alert('Failed to load animation: ' + err.message);
        }
      };
      reader.readAsArrayBuffer(f);
    });
  }
  var hsSlider = document.getElementById('heightScale');
  var hsVal = document.getElementById('heightScaleVal');
  if (hsSlider && hsVal) {
    hsSlider.addEventListener('input', function() {
      heightScale = parseFloat(hsSlider.value);
      hsVal.textContent = heightScale.toFixed(2);
      if (water.anim) {
        water.playFrame(water.anim.idx || 0, heightScale);
        renderer.updateCaustics(water);
        draw();
      }
    });
  }

  /* Play/pause button: mirrors the spacebar handler */
  var playPauseBtn = document.getElementById('playPauseBtn');
  function refreshPlayPause() {
    if (playPauseBtn) playPauseBtn.textContent = paused ? 'Play' : 'Pause';
  }
  if (playPauseBtn) {
    playPauseBtn.addEventListener('click', function() {
      paused = !paused;
      refreshPlayPause();
    });
  }

  /* Time scrub slider: when the user drags, jump animation time to that
   * position. When animation plays, the slider position is updated to
   * reflect current animTime (see update() / draw()). */
  var timeScrub = document.getElementById('timeScrub');
  var timeLabel = document.getElementById('timeLabel');
  var scrubbing = false;
  if (timeScrub) {
    timeScrub.addEventListener('input', function() {
      if (!water.anim || water.anim.nFrames <= 1) return;
      scrubbing = true;
      var period = water.anim.period_s > 0 ? water.anim.period_s : 1.0;
      animTime = parseFloat(timeScrub.value) * period;
      var idx = Math.floor((animTime / period) * water.anim.nFrames) % water.anim.nFrames;
      water.anim.idx = idx;
      water.playFrame(idx, heightScale);
      renderer.updateCaustics(water);
      draw();
      if (timeLabel) {
        timeLabel.textContent = 't = ' + animTime.toFixed(2) + ' / ' +
                                period.toFixed(2) + ' s';
      }
    });
    timeScrub.addEventListener('change', function() {
      scrubbing = false;
    });
  }

  document.getElementById('loading').innerHTML = '';
  onresize();

  var requestAnimationFrame =
    window.requestAnimationFrame ||
    window.webkitRequestAnimationFrame ||
    function(callback) { setTimeout(callback, 0); };

  var prevTime = new Date().getTime();
  function animate() {
    var nextTime = new Date().getTime();
    if (!paused) {
      update((nextTime - prevTime) / 1000);
      draw();
    }
    prevTime = nextTime;
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  window.onresize = onresize;

  var prevHit;
  var planeNormal;
  var mode = -1;
  var MODE_ADD_DROPS = 0;
  var MODE_MOVE_SPHERE = 1;
  var MODE_ORBIT_CAMERA = 2;

  var oldX, oldY;

  function startDrag(x, y) {
    /* Water_v2: always orbit; sphere + drop modes removed */
    oldX = x;
    oldY = y;
    mode = MODE_ORBIT_CAMERA;
  }

  function duringDrag(x, y) {
    switch (mode) {
      case MODE_ADD_DROPS: {
        var tracer = new GL.Raytracer();
        var ray = tracer.getRayForPixel(x * ratio, y * ratio);
        var pointOnPlane = tracer.eye.add(ray.multiply(-tracer.eye.y / ray.y));
        water.addDrop(pointOnPlane.x, pointOnPlane.z, 0.03, 0.01);
        if (paused) {
          water.updateNormals();
          renderer.updateCaustics(water);
        }
        break;
      }
      case MODE_MOVE_SPHERE: {
        var tracer = new GL.Raytracer();
        var ray = tracer.getRayForPixel(x * ratio, y * ratio);
        var t = -planeNormal.dot(tracer.eye.subtract(prevHit)) / planeNormal.dot(ray);
        var nextHit = tracer.eye.add(ray.multiply(t));
        center = center.add(nextHit.subtract(prevHit));
        center.x = Math.max(radius - 1, Math.min(1 - radius, center.x));
        center.y = Math.max(radius - 1, Math.min(10, center.y));
        center.z = Math.max(radius - 1, Math.min(1 - radius, center.z));
        prevHit = nextHit;
        if (paused) renderer.updateCaustics(water);
        break;
      }
      case MODE_ORBIT_CAMERA: {
        angleY -= x - oldX;
        angleX -= y - oldY;
        angleX = Math.max(-89.999, Math.min(89.999, angleX));
        break;
      }
    }
    oldX = x;
    oldY = y;
    if (paused) draw();
  }

  function stopDrag() {
    mode = -1;
  }

  function isHelpElement(element) {
    /* Any element inside the controls or caustic-view panels is exempted
     * from the camera-drag handler so the controls themselves work. */
    return element === controls || element === causticView
        || element.parentNode && isHelpElement(element.parentNode);
  }

  document.onmousedown = function(e) {
    if (!isHelpElement(e.target)) {
      e.preventDefault();
      startDrag(e.pageX, e.pageY);
    }
  };

  document.onmousemove = function(e) {
    duringDrag(e.pageX, e.pageY);
  };

  document.onmouseup = function() {
    stopDrag();
  };

  document.ontouchstart = function(e) {
    if (e.touches.length === 1 && !isHelpElement(e.target)) {
      e.preventDefault();
      startDrag(e.touches[0].pageX, e.touches[0].pageY);
    }
  };

  document.ontouchmove = function(e) {
    if (e.touches.length === 1) {
      duringDrag(e.touches[0].pageX, e.touches[0].pageY);
    }
  };

  document.ontouchend = function(e) {
    if (e.touches.length == 0) {
      stopDrag();
    }
  };

  document.onkeydown = function(e) {
    if (e.which == ' '.charCodeAt(0)) {
      paused = !paused;
      refreshPlayPause();
    }
    else if (e.which == 'G'.charCodeAt(0)) useSpherePhysics = !useSpherePhysics;
    else if (e.which == 'L'.charCodeAt(0) && paused) draw();
  };

  var frame = 0;

  function update(seconds) {
    if (seconds > 1) return;
    frame += seconds * 2;

    /* Water_v2: replace simulation with animation playback. If no animation
     * loaded, do nothing (the water just stays however it was last set). */
    if (!water.anim) return;
    if (water.anim.nFrames <= 1) return;   /* static — already uploaded on load */
    if (scrubbing) return;                 /* user is dragging the time slider */

    animTime += seconds;
    var period = water.anim.period_s > 0 ? water.anim.period_s : 1.0;
    var phase = (animTime % period) / period;
    var idx = Math.floor(phase * water.anim.nFrames) % water.anim.nFrames;
    if (idx !== water.anim.idx) {
      water.anim.idx = idx;
      water.playFrame(idx, heightScale);
      renderer.updateCaustics(water);
    }
    /* Reflect time on the UI */
    if (timeScrub) timeScrub.value = phase;
    if (timeLabel) {
      timeLabel.textContent = 't = ' + (phase * period).toFixed(2) +
                              ' / ' + period.toFixed(2) + ' s';
    }
  }

  function draw() {
    // Change the light direction to the camera look vector when the L key is pressed
    if (GL.keys.L) {
      renderer.lightDir = GL.Vector.fromAngles((90 - angleY) * Math.PI / 180, -angleX * Math.PI / 180);
      if (paused) renderer.updateCaustics(water);
    }

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.loadIdentity();
    gl.translate(0, 0, -4);
    gl.rotate(-angleX, 1, 0, 0);
    gl.rotate(-angleY, 0, 1, 0);
    gl.translate(0, 0.5, 0);

    gl.enable(gl.DEPTH_TEST);
    renderer.sphereCenter = center;
    renderer.sphereRadius = radius;   /* radius=0 → sphere effectively invisible */
    renderer.renderCube();
    renderer.renderWater(water, cubemap);
    if (radius > 0) renderer.renderSphere();
    gl.disable(gl.DEPTH_TEST);

    /* Top-down caustic preview, drawn AFTER the main 3D scene so the
     * readPixels uses the freshly-baked causticTex. The function
     * early-returns when the panel is collapsed. */
    updateCausticPreview();
  }
};
