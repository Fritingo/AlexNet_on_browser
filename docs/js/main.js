'use strict';



const videoElement = document.querySelector('video');
const videoSelect = document.querySelector('select#videoSource');
const selectors = [videoSelect];

const canvas = document.querySelector('canvas');

// ------------------load file-------------------------

var fileElem = document.getElementById("fileElem");
fileElem.addEventListener('change',handlefile,false);




// ------------------device setting----------------------
function gotDevices(deviceInfos) {
  // Handles being called several times to update labels. Preserve values.
  const values = selectors.map(select => select.value);
  selectors.forEach(select => {
    while (select.firstChild) {
      select.removeChild(select.firstChild);
    }
  });
  for (let i = 0; i !== deviceInfos.length; ++i) {
    const deviceInfo = deviceInfos[i];
    const option = document.createElement('option');
    option.value = deviceInfo.deviceId;
    if (deviceInfo.kind === 'videoinput') {
      option.text = deviceInfo.label || `camera ${videoSelect.length + 1}`;
      videoSelect.appendChild(option);
    } else {
      console.log('Some other kind of source/device: ', deviceInfo);
    }
  }
  selectors.forEach((select, selectorIndex) => {
    if (Array.prototype.slice.call(select.childNodes).some(n => n.value === values[selectorIndex])) {
      select.value = values[selectorIndex];
    }
  });
}

navigator.mediaDevices.enumerateDevices().then(gotDevices).catch(handleError);
// ------------------device setting----------------------


// Attach audio output device to video element using device/sink ID.
function attachSinkId(element, sinkId) {
  if (typeof element.sinkId !== 'undefined') {
    element.setSinkId(sinkId)
        .then(() => {
          console.log(`Success, audio output device attached: ${sinkId}`);
        })
        .catch(error => {
          let errorMessage = error;
          if (error.name === 'SecurityError') {
            errorMessage = `You need to use HTTPS for selecting audio output device: ${error}`;
          }
          console.error(errorMessage);
          // Jump back to first output device in the list as it's the default.
          audioOutputSelect.selectedIndex = 0;
        });
  } else {
    console.warn('Browser does not support output device selection.');
  }
}

function changeAudioDestination() {
  const audioDestination = audioOutputSelect.value;
  attachSinkId(videoElement, audioDestination);
}

function gotStream(stream) {
  window.stream = stream; // make stream available to console
  videoElement.srcObject = stream;
  // Refresh button list in case labels have become available
  return navigator.mediaDevices.enumerateDevices();
}

function handleError(error) {
  console.log('navigator.MediaDevices.getUserMedia error: ', error.message, error.name);
}

function start() {
  if (window.stream) {
    window.stream.getTracks().forEach(track => {
      track.stop();
    });
  }
  
  const videoSource = videoSelect.value;
  const constraints = {
    audio: false,
    video: {deviceId: videoSource ? {exact: videoSource} : undefined,
            "width": {
              "min": "640",
              "max": "640"
          },
          "height": {
              "min": "480",
              "max": "480"
  }},
    
  };
  navigator.mediaDevices.getUserMedia(constraints).then(gotStream).then(gotDevices).catch(handleError);
}


// ---------------- canvas ---------------------
const ctx = canvas.getContext('2d');

// ---------------- file to canvas--------------
function handlefile(e){
  var reader = new FileReader();
  reader.onload = function(event){
      var img = new Image();
      img.onload = function(){
          // canvas.width = img.width;
          // canvas.height = img.height;
          canvas.width = 640;
          canvas.height = 480;

          ctx.drawImage(img,0,0,640,480);
      }
      img.src = event.target.result;
  }
  reader.readAsDataURL(e.target.files[0]);     
}

function press(){
  console.log('press');
  
}


//----------- take picture ------------------
function take_picture(){
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
}




//----------------canvas---------------

// ------------model-----------------------------------------


// ------------------- Load our model.-----------------------

var sess;
const init_model = async ()=>{
  console.log("Loading Session and model")
  sess = new onnx.InferenceSession({backendHint: 'webgl'});
  await sess.loadModel("./cifar100_Alexnet.onnx");
  console.log("Done")
}

init_model();


async function updatePredictions() {
  
  const imgData = ctx.getImageData(0, 0, 128, 128);//red, green, blue, alpha (0 ~ 255)
  const input = new onnx.Tensor(new Float32Array(imgData.data), "float32");
 
  const outputMap = await sess.run([input]);
  const outputTensor = outputMap.values().next().value;
  const predictions = outputTensor.data;
  const maxPrediction = predictions.indexOf(Math.max(...predictions));
  let ans = JSON.stringify(predictions[0])
  document.getElementById("ans").innerHTML = classes[maxPrediction];
  // console.log('testing')
  
}





videoSelect.onchange = start;

start();

// setInterval(show, 1000 / 100);
