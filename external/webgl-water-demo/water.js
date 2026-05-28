/*
 * WebGL Water
 * https://madebyevan.com/webgl-water/
 *
 * Copyright 2011 Evan Wallace
 * Released under the MIT license
 */

// The data in the texture is (position.y, velocity.y, normal.x, normal.z)
function Water() {
  var vertexShader = '\
    varying vec2 coord;\
    void main() {\
      coord = gl_Vertex.xy * 0.5 + 0.5;\
      gl_Position = vec4(gl_Vertex.xyz, 1.0);\
    }\
  ';
  this.plane = GL.Mesh.plane();
  if (!GL.Texture.canUseFloatingPointTextures()) {
    throw new Error('This demo requires the OES_texture_float extension');
  }
  var filter = GL.Texture.canUseFloatingPointLinearFiltering() ? gl.LINEAR : gl.NEAREST;
  this.textureA = new GL.Texture(256, 256, { type: gl.FLOAT, filter: filter });
  this.textureB = new GL.Texture(256, 256, { type: gl.FLOAT, filter: filter });
  if ((!this.textureA.canDrawTo() || !this.textureB.canDrawTo()) && GL.Texture.canUseHalfFloatingPointTextures()) {
    filter = GL.Texture.canUseHalfFloatingPointLinearFiltering() ? gl.LINEAR : gl.NEAREST;
    this.textureA = new GL.Texture(256, 256, { type: gl.HALF_FLOAT_OES, filter: filter });
    this.textureB = new GL.Texture(256, 256, { type: gl.HALF_FLOAT_OES, filter: filter });
  }
  this.dropShader = new GL.Shader(vertexShader, '\
    const float PI = 3.141592653589793;\
    uniform sampler2D texture;\
    uniform vec2 center;\
    uniform float radius;\
    uniform float strength;\
    varying vec2 coord;\
    void main() {\
      /* get vertex info */\
      vec4 info = texture2D(texture, coord);\
      \
      /* add the drop to the height */\
      float drop = max(0.0, 1.0 - length(center * 0.5 + 0.5 - coord) / radius);\
      drop = 0.5 - cos(drop * PI) * 0.5;\
      info.r += drop * strength;\
      \
      gl_FragColor = info;\
    }\
  ');
  this.updateShader = new GL.Shader(vertexShader, '\
    uniform sampler2D texture;\
    uniform vec2 delta;\
    varying vec2 coord;\
    void main() {\
      /* get vertex info */\
      vec4 info = texture2D(texture, coord);\
      \
      /* calculate average neighbor height */\
      vec2 dx = vec2(delta.x, 0.0);\
      vec2 dy = vec2(0.0, delta.y);\
      float average = (\
        texture2D(texture, coord - dx).r +\
        texture2D(texture, coord - dy).r +\
        texture2D(texture, coord + dx).r +\
        texture2D(texture, coord + dy).r\
      ) * 0.25;\
      \
      /* change the velocity to move toward the average */\
      info.g += (average - info.r) * 2.0;\
      \
      /* attenuate the velocity a little so waves do not last forever */\
      info.g *= 0.995;\
      \
      /* move the vertex along the velocity */\
      info.r += info.g;\
      \
      gl_FragColor = info;\
    }\
  ');
  this.normalShader = new GL.Shader(vertexShader, '\
    uniform sampler2D texture;\
    uniform vec2 delta;\
    varying vec2 coord;\
    void main() {\
      /* get vertex info */\
      vec4 info = texture2D(texture, coord);\
      \
      /* update the normal */\
      vec3 dx = vec3(delta.x, texture2D(texture, vec2(coord.x + delta.x, coord.y)).r - info.r, 0.0);\
      vec3 dy = vec3(0.0, texture2D(texture, vec2(coord.x, coord.y + delta.y)).r - info.r, delta.y);\
      info.ba = normalize(cross(dy, dx)).xz;\
      \
      gl_FragColor = info;\
    }\
  ');
  this.sphereShader = new GL.Shader(vertexShader, '\
    uniform sampler2D texture;\
    uniform vec3 oldCenter;\
    uniform vec3 newCenter;\
    uniform float radius;\
    varying vec2 coord;\
    \
    float volumeInSphere(vec3 center) {\
      vec3 toCenter = vec3(coord.x * 2.0 - 1.0, 0.0, coord.y * 2.0 - 1.0) - center;\
      float t = length(toCenter) / radius;\
      float dy = exp(-pow(t * 1.5, 6.0));\
      float ymin = min(0.0, center.y - dy);\
      float ymax = min(max(0.0, center.y + dy), ymin + 2.0 * dy);\
      return (ymax - ymin) * 0.1;\
    }\
    \
    void main() {\
      /* get vertex info */\
      vec4 info = texture2D(texture, coord);\
      \
      /* add the old volume */\
      info.r += volumeInSphere(oldCenter);\
      \
      /* subtract the new volume */\
      info.r -= volumeInSphere(newCenter);\
      \
      gl_FragColor = info;\
    }\
  ');
}

Water.prototype.addDrop = function(x, y, radius, strength) {
  var this_ = this;
  this.textureB.drawTo(function() {
    this_.textureA.bind();
    this_.dropShader.uniforms({
      center: [x, y],
      radius: radius,
      strength: strength
    }).draw(this_.plane);
  });
  this.textureB.swapWith(this.textureA);
};

Water.prototype.moveSphere = function(oldCenter, newCenter, radius) {
  var this_ = this;
  this.textureB.drawTo(function() {
    this_.textureA.bind();
    this_.sphereShader.uniforms({
      oldCenter: oldCenter,
      newCenter: newCenter,
      radius: radius
    }).draw(this_.plane);
  });
  this.textureB.swapWith(this.textureA);
};

Water.prototype.stepSimulation = function() {
  var this_ = this;
  this.textureB.drawTo(function() {
    this_.textureA.bind();
    this_.updateShader.uniforms({
      delta: [1 / this_.textureA.width, 1 / this_.textureA.height]
    }).draw(this_.plane);
  });
  this.textureB.swapWith(this.textureA);
};

Water.prototype.updateNormals = function() {
  var this_ = this;
  this.textureB.drawTo(function() {
    this_.textureA.bind();
    this_.normalShader.uniforms({
      delta: [1 / this_.textureA.width, 1 / this_.textureA.height]
    }).draw(this_.plane);
  });
  this.textureB.swapWith(this.textureA);
};

/* ---------- Animation playback (added for Water_v2 caustic project) ----------
 *
 * loadAnimation(arrayBuffer): parse a binary surface-η animation produced by
 * notebooks/export_3d_animation.py, store all frames, and store geometric
 * parameters (Lx, depth).
 *
 * playFrame(frameIdx, scale): upload the chosen frame's η to textureA's R
 * channel and re-compute the normals via the existing normalShader.
 *
 * The binary layout:
 *   [u32 0xDEADBEEF][u32 nFrames][u32 nx][u32 ny][f32 period_s]
 *   [f32 Lx][f32 depth][f32 etaMax][f32 * nFrames*nx*ny]
 */
Water.prototype.loadAnimation = function(arrayBuffer) {
  var dv = new DataView(arrayBuffer);
  var magic = dv.getUint32(0, true);
  if (magic !== 0xDEADBEEF) throw new Error('Bad animation magic: ' + magic.toString(16));
  this.anim = {
    nFrames:  dv.getUint32(4, true),
    nx:       dv.getUint32(8, true),
    ny:       dv.getUint32(12, true),
    period_s: dv.getFloat32(16, true),
    Lx:       dv.getFloat32(20, true),
    depth:    dv.getFloat32(24, true),
    etaMax:   dv.getFloat32(28, true)
  };
  var dataStart = 32;
  var frameSize = this.anim.nx * this.anim.ny;
  var heightsBytes = this.anim.nFrames * frameSize * 4;
  this.anim.frames = new Float32Array(arrayBuffer, dataStart,
                                       this.anim.nFrames * frameSize);
  /* Optional: 2 extra Float32 channels per voxel appended after the heights
   * block carry analytical normals (info.b, info.a) computed at export time
   * from the cos basis. If present, playFrame uploads them directly and
   * skips normalShader, avoiding the 4-neighbor finite-diff low-pass that
   * Wallace's stencil would otherwise apply to our surface gradients. */
  var normalsBytes = this.anim.nFrames * frameSize * 8;   /* 2 floats */
  if (arrayBuffer.byteLength >= dataStart + heightsBytes + normalsBytes) {
    this.anim.normals = new Float32Array(arrayBuffer,
                                          dataStart + heightsBytes,
                                          this.anim.nFrames * frameSize * 2);
  } else {
    this.anim.normals = null;
  }
  this.anim.idx = 0;
  console.log('Loaded animation:', this.anim.nFrames, 'frames at',
              this.anim.nx + 'x' + this.anim.ny,
              'Lx=' + this.anim.Lx + 'm  depth=' + this.anim.depth + 'm  |η|_max=' +
              this.anim.etaMax.toExponential(3) + 'm  normals=' +
              (this.anim.normals ? 'yes (analytical)' : 'no (Wallace FD)'));
  return this.anim;
};

/* Upload a specific animation frame to textureA, then run normalShader.
 * scale lets us amplify height variations for visual punch (caustic intensity
 * scales with surface curvature, which scales with amplitude — small physical
 * heights produce subtle caustics by design, but we may want to crank for the
 * 3D demo).
 */
Water.prototype.playFrame = function(frameIdx, scale) {
  if (!this.anim) return;
  scale = (scale === undefined) ? 1.0 : scale;
  var nx = this.anim.nx, ny = this.anim.ny;
  if (nx !== this.textureA.width || ny !== this.textureA.height) {
    throw new Error('Animation grid ' + nx + 'x' + ny +
                    ' does not match water texture ' +
                    this.textureA.width + 'x' + this.textureA.height);
  }
  var frame = this.anim.frames.subarray(frameIdx * nx * ny,
                                         (frameIdx + 1) * nx * ny);
  /* Build RGBA float buffer: r = η * scale, g = 0, b/a = normals (or 0).
   * If analytical normals are present in the .bin, write them directly
   * into b/a and skip the updateNormals() shader pass below. Otherwise
   * leave b/a = 0 and run normalShader (Wallace's 4-neighbor FD). */
  var rgba = new Float32Array(nx * ny * 4);
  var hasNormals = !!this.anim.normals;
  if (hasNormals) {
    /* Normals are unit vectors (independent of height-scale), so we
     * don't multiply by scale. NOTE: this means analytical normals are
     * only physically consistent at scale=1.0; for diagnostic comparison
     * against our optimization keep the slider at 1.0. */
    var nview = this.anim.normals.subarray(frameIdx * nx * ny * 2,
                                             (frameIdx + 1) * nx * ny * 2);
    for (var i = 0; i < nx * ny; i++) {
      rgba[i * 4]     = frame[i] * scale;
      /* g (velocity) stays 0 — only meaningful for live simulation */
      rgba[i * 4 + 2] = nview[i * 2];        /* info.b = n_x  (unscaled) */
      rgba[i * 4 + 3] = nview[i * 2 + 1];    /* info.a = n_z  (unscaled) */
    }
  } else {
    for (var i = 0; i < nx * ny; i++) {
      rgba[i * 4] = frame[i] * scale;
    }
  }
  this.textureA.bind();
  /* Match the texture's actual type — Wallace falls back to HALF_FLOAT_OES
   * on browsers without full FLOAT support. */
  if (this.textureA.type === gl.FLOAT) {
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, nx, ny, gl.RGBA, gl.FLOAT, rgba);
  } else if (this.textureA.type === gl.HALF_FLOAT_OES) {
    /* Convert Float32 → Uint16 half-floats */
    var half = new Uint16Array(rgba.length);
    var f32 = new Float32Array(1);
    var u32 = new Uint32Array(f32.buffer);
    for (var k = 0; k < rgba.length; k++) {
      f32[0] = rgba[k];
      var x = u32[0];
      var sign = (x >> 16) & 0x8000;
      var mantissa = x & 0x007FFFFF;
      var exp = ((x >> 23) & 0xFF) - 127 + 15;
      if (exp <= 0) {
        half[k] = sign;
      } else if (exp >= 31) {
        half[k] = sign | 0x7C00;
      } else {
        half[k] = sign | (exp << 10) | (mantissa >> 13);
      }
    }
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, nx, ny, gl.RGBA, gl.HALF_FLOAT_OES, half);
  } else {
    throw new Error('Unsupported texture type for animation upload: 0x' +
                    this.textureA.type.toString(16));
  }
  /* Only run the 4-neighbor FD normalShader when we don't have
   * analytical normals from the exporter. With analytical normals,
   * info.b and info.a were just written directly above and would be
   * overwritten if we ran the shader. */
  if (!hasNormals) {
    this.updateNormals();
  }
};
