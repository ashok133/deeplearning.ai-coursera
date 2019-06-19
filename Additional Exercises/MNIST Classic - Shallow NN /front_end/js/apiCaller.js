var apiCallerApp = new Vue({
	el: '#apiCallerDiv',
	data : {
		// possibe
		prediction_text: 'X'
		},
	  methods: {
      detectSimilarImages() {
				// document.getElementById('image-card').style.visibility = 'visible';
        var files = document.getElementById('src-img').files;
        if (files.length > 0) {
          var reader = new FileReader();
          reader.readAsDataURL(files[0]);
          reader.onload = function () {
            var output = document.getElementById('src-decoded');
						output.style.visibility = 'visible';
      			output.src = reader.result;
            apiCallerApp.fetchSimilarImageResults(reader.result.split(',')[1]);
            // console.log("sadjahgsdagsdgahsdgahdgs",reader.result.split(',')[1]);
          };
					// reader.readAsDataURL(event.target.files[0]);
          reader.onerror = function (error) {
            console.log('Error: ', error);
          };
        }
        else {
          console.error('Couldn\'t upload the file');
				}
      },
      fetchSimilarImageResults(src_b64) {
				fetch('https://sign-classifier.appspot.com/predict_sign', {
					body : JSON.stringify({
    				'base64_str' : src_b64
  				    }),
					     mode: "cors", // no-cors, cors, *same-origin
					     headers: {
    				    'Accept': 'application/json, text/plain, */*',
    				    'Content-Type': 'application/json; charset=utf-8'
  				    },
					     method: "POST"
				    }
			   )
				 .then((res) => {
           return res.json();
      	})
					.then (json => {
					 	// console.log(json.responses[0].landmarkAnnotations[0].mid)
            console.log(json);
 						apiCallerApp.prediction_text = json['prediction'];
 				})
  			.catch( function(err){
  				console.log(err)
  			})
			},
			fetchBase64FromLoc(canvas, img_id) {
				var c = document.getElementById('img-canvas');
				var ctx = c.getContext("2d");
				var img = document.getElementById('sample-img-one');
				ctx.drawImage(img, 1, 1);
				return (c.toDataURL('image/jpeg'));
			},
			fetchSampleResults(canvas, img_id) {
				var b64 = apiCallerApp.fetchBase64FromLoc(canvas, img_id);
				apiCallerApp.fetchSimilarImageResults(b64);
			}
	  },

    created: function() {
      // --
    }
	})
