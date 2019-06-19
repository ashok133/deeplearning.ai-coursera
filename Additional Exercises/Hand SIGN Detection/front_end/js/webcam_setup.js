Webcam.set({
  width: 320,
  height: 240,
  image_format: 'jpeg',
  jpeg_quality: 90
 });
 Webcam.attach( '#sign-image' );

function take_snapshot() {
 // take snapshot and get image data
 Webcam.snap( function(data_uri) {
  // display results in page
  document.getElementById('src-decoded').innerHTML =
  '<img src="'+data_uri+'"/>';
  } );
}
