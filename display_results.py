from io import BytesIO
import base64
import logging
import numpy as np
import IPython.display
from string import Template

from lucid.misc.io.serialize_array import serialize_array, array_to_jsbuffer


def _image_url(array, fmt='png', mode="data", quality=90, domain=None):
    """Create a data URL representing an image from a PIL.Image.

    Args:
        image: a numpy
        mode: presently only supports "data" for data URL

    Returns:
        URL representing image
    """
    # TODO: think about supporting saving to CNS, potential params: cns, name
    # TODO: think about saving to Cloud Storage
    supported_modes = ("data")
    if mode not in supported_modes:
        message = "Unsupported mode '%s', should be one of '%s'."
        raise ValueError(message, mode, supported_modes)

    image_data = serialize_array(array, fmt=fmt, quality=quality)
    base64_byte_string = base64.b64encode(image_data).decode('ascii')
    return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


def show_textured_mesh(path, mesh, texture, background='0xffffff'):
  texture_data_url = _image_url(texture, fmt='jpeg', quality=90)

  code = Template('''
  <input id="unfoldBox" type="checkbox" class="control">Unfold</input>
  <input id="shadeBox" type="checkbox" class="control">Shade</input>

  <script src="https://cdn.rawgit.com/mrdoob/three.js/r89/build/three.min.js"></script>
  <script src="https://cdn.rawgit.com/mrdoob/three.js/r89/examples/js/controls/OrbitControls.js"></script>

  <script type="x-shader/x-vertex" id="vertexShader">
    uniform float viewAspect;
    uniform float unfolding_perc;
    uniform float shadeFlag;
    varying vec2 text_coord;
    varying float shading;
    void main () {
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      vec4 plane_position = vec4((uv.x*2.0-1.0)/viewAspect, (uv.y*2.0-1.0), 0, 1);
      gl_Position = mix(gl_Position, plane_position, unfolding_perc);

      //not normalized on purpose to simulate the rotation
      shading = 1.0;
      if (shadeFlag > 0.5) {
        vec3 light_vector = mix(normalize(cameraPosition-position), normal, unfolding_perc);
        shading = dot(normal, light_vector);
      }

      text_coord = uv;
    }
  </script>

  <script type="x-shader/x-fragment" id="fragmentShader">
    uniform float unfolding_perc;
    varying vec2  text_coord;
    varying float shading;
    uniform sampler2D texture;

    void main() {
      gl_FragColor = texture2D(texture, text_coord);
      gl_FragColor.rgb *= shading;
    }
  </script>

  <script>
  "use strict";

  const el = id => document.getElementById(id);

  const unfoldDuration = 1000.0;
  var camera, scene, renderer, controls, material;
  var unfolded = false;
  var unfoldStart = -unfoldDuration*10.0;

  init();
  animate(0.0);

  function init() {
    var width = 800, height = 600;

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.z = 3.3;
    scene.add(camera);

    controls = new THREE.OrbitControls( camera );

    var geometry = new THREE.BufferGeometry();
    geometry.addAttribute( 'position', new THREE.BufferAttribute($verts, 3 ) );
    geometry.addAttribute( 'uv', new THREE.BufferAttribute($uvs, 2) );
    geometry.setIndex(new THREE.BufferAttribute($faces, 1 ));
    geometry.computeVertexNormals();

    var texture = new THREE.TextureLoader().load('$tex_data_url', update);
    material = new THREE.ShaderMaterial( {
      uniforms: {
        viewAspect: {value: width/height},
        unfolding_perc: { value: 0.0 },
        shadeFlag: { value: 0.0 },
        texture: { type: 't', value: texture },
      },
      side: THREE.DoubleSide,
      vertexShader: el( 'vertexShader' ).textContent,
      fragmentShader: el( 'fragmentShader' ).textContent
    });

    var mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    scene.background = new THREE.Color( $background );

    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(width, height);
    document.body.appendChild(renderer.domElement);

    // render on change only
    controls.addEventListener('change', function() {
      // fold mesh back if user wants to interact
      el('unfoldBox').checked = false;
      update();
    });
    document.querySelectorAll('.control').forEach(e=>{
      e.addEventListener('change', update);
    });
  }

  function update() {
    requestAnimationFrame(animate);
  }

  function ease(x) {
    x = Math.min(Math.max(x, 0.0), 1.0);
    return x*x*(3.0 - 2.0*x);
  }

  function animate(time) {
    var unfoldFlag = el('unfoldBox').checked;
    if (unfolded != unfoldFlag) {
      unfolded = unfoldFlag;
      unfoldStart = time - Math.max(unfoldStart+unfoldDuration-time, 0.0);
    }
    var unfoldTime = (time-unfoldStart) / unfoldDuration;
    if (unfoldTime < 1.0) {
      update();
    }
    var unfoldVal = ease(unfoldTime);
    unfoldVal = unfolded ? unfoldVal : 1.0 - unfoldVal;
    material.uniforms.unfolding_perc.value = unfoldVal;

    material.uniforms.shadeFlag.value = el('shadeBox').checked ? 1.0 : 0.0;
    controls.update();
    renderer.render(scene, camera);
  }
  </script>
  ''').substitute(
      verts = array_to_jsbuffer(mesh['position'].ravel()),
      uvs = array_to_jsbuffer(mesh['uv'].ravel()),
      faces = array_to_jsbuffer(np.uint32(mesh['face'].ravel())),
      tex_data_url = texture_data_url,
      background = background,
  )
  with open(path, "w") as file:
    file.write(code)
