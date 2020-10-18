var firebaseConfig = {
  apiKey: "AIzaSyDTwwK3SlpVUHnex6xCkEbvKIZSGOCYghY",
  authDomain: "speechmania-b547e.firebaseapp.com",
  databaseURL: "https://speechmania-b547e.firebaseio.com",
  projectId: "speechmania-b547e",
  storageBucket: "speechmania-b547e.appspot.com",
  messagingSenderId: "694335410890",
  appId: "1:694335410890:web:8d333835b9433c781b9a33",
  measurementId: "G-GZFRM55DSZ"
};
  /*// Initialize Firebase
  firebase.initializeApp(firebaseConfig);
  //firebase.analytics();
  firebase.database().ref().on('value', function(snapshot){
    data = snapshot.val()
     var Accuracy=(data.Chair_Accuracy);
     //var y=(Data.Y);
     console.log(data);
    console.log(Accuracy);
    //console.log(y);
});



var span = document.getElementById("percent1"); 
console.log(span);*/
firebase.initializeApp(firebaseConfig);
var ref = firebase.database().ref()/*.child('speechmania-b547e')*/;
 
ref.on("value", function(snapshot) {
  var Data=snapshot.val();
  var chair_accuracy=(Data.x);//as a variable y in line chart 1 and variable in bie chart word1
  var door_accuracy=(Data.Door_Accuracy);//as a variable y in line chart 2
console.log(Data);
console.log(chair_accuracy);
console.log(door_accuracy);
  
});
















if (actual_word==predicted_word=='chair')
		  {
		
			y.push(Data.Chair_Accuracy);
			x.push(String(date.getDate())+'-'+String(date.getMonth()+1)+'-'+String(date.getFullYear())/*+'-'+String(i)*/);
			createChart(x, y);
			old_chair_accuracy=chair_accuracy;
			i=i+1;
		  }
else{
			  y.push(0);
			  x.push(String(date.getDate())+'-'+String(date.getMonth()+1)+'-'+String(date.getFullYear())/*+'-'+String(i)*/);
			createChart(x, y);
		  }
 
if (actual_word==predicted_word=='door')
		  {
			z.push(Data.Door_Accuracy);
			x.push(String(date.getDate())+'-'+String(date.getMonth()+1)+'-'+String(date.getFullYear())/*+'-'+String(i)*/);
			createChart(x, z);
			old_door_accuracy=Door_accuracy;
			i=i+1;
		  }
else{
			  z.push(0);
			  x.push(String(date.getDate())+'-'+String(date.getMonth()+1)+'-'+String(date.getFullYear())/*+'-'+String(i)*/);
			createChart(x, z);
		  }






































/*var randomScalingFactor = function(){ return Math.round(Math.random()*1000)};
var date = new Date();
//var Accuracy=0;
//console.log(date.getDate());
var firebaseConfig = {
    apiKey: "AIzaSyDTwwK3SlpVUHnex6xCkEbvKIZSGOCYghY",
    authDomain: "speechmania-b547e.firebaseapp.com",
    databaseURL: "https://speechmania-b547e.firebaseio.com",
    projectId: "speechmania-b547e",
    storageBucket: "speechmania-b547e.appspot.com",
    messagingSenderId: "694335410890",
    appId: "1:694335410890:web:8d333835b9433c781b9a33",
    measurementId: "G-GZFRM55DSZ"
  };
  var oldAccuracy1=0;
  var oldAccuracy2=0;
  var chart1 = document.getElementById("line-chart").getContext("2d");
  
  var y=[];
  var ynew=[];
  var z=[];
  var x=[] /*Array.from(Array(10000).keys()).map(i => 0 + i *5 )*/;
  i=1;
  // Initialize Firebase
  firebase.initializeApp(firebaseConfig);
 //piechart(Accuracy);
  /*var msglist = document.getElementById("easypiechart-blue");
  //firebase.analytics();
  var ref = firebase.database().ref().child('first-trial-60980');
 
  ref.on("value", function(snapshot) {
     var Data=snapshot.val();
     var Accuracy=(Data.accuracy);//as a variable y in line chart 1 and variable in bie chart word1
	 var Accuracy2=(Data.accuracy2);//as a variable y in line chart 2

	 //for bie chart 
	 /*var msglist = document.getElementById("easypiechart-blue");
	var show = msglist.getAttribute("data-percent");
	  console.log(show);
	msglist.removeAttribute("data-percent");
	var show = msglist.getAttribute("data-percent");
	console.log(show);
	msglist.setAttribute("data-percent", String(Accuracy));
	var show = msglist.getAttribute("data-percent");
	console.log(show);
	var value = 50;*/
	//$("#easypiechart-blue").attr("data-percent", Accuracy.toString());
	//var theValue = document.getElementById('easypiechart-blue').getElementsByClassName('percent')[0].innerHTML;
	








   // piechart(Accuracy);
	/*document.getElementById('easypiechart-blue').getElementsByClassName('percent')[0].innerHTML=String(Accuracy)+'%';
	







	// for the data percent changing//this method affect only on js but don't return the value to the html

	//var msglist = document.getElementById("easypiechart-blue");
    //console.log(msglist);
	
    
// i will try this method


	//document.getElementById("easypiechart-blue").setAttribute("data-percent","50");
	//var x = document.getElementsByTagName("data-percent");
	//$('.easypiechart').data('data-percent', 50);
	//var e = document.getElementById("easypiechart-blue"); //Get the element
    //e.setAttribute("id", "div3"); //Change id to div3

    //$("#easypiechart-blue").attr("data-percent", Accuracy.toString());
	



     console.log(Data);
	console.log(Accuracy);
	console.log(Accuracy);
    if ((Accuracy!=oldAccuracy1)||(Accuracy2!=oldAccuracy2))
    {
	  y.push(Data.accuracy);
	  z.push(Data.accuracy2)
	  x.push(String(date.getDate())+'-'+String(date.getMonth()+1)+'-'+String(date.getFullYear())/*+'-'+String(i));*/
	  //x.push('Trial'+String(i));
	  //console.log(String(i));
	  //x.push(date);

	  
	  
	 /* createChart(x, y);
	// piechart(y);
	  oldAccuracy=Accuracy;
	  oldAccuracy2=Accuracy2;
	  i=i+1;
    }
});

var lineChartData = {
	labels : x,
	datasets : [
		{
			label: "the First word",
			fillColor : "rgba(220,220,220,0.2)",
			strokeColor : "rgba(220,220,220,1)",
			pointColor : "rgba(220,220,220,1)",
			pointStrokeColor : "#fff",
			pointHighlightFill : "#fff",
			pointHighlightStroke : "rgba(220,220,220,1)",
			data : y
		},
		{
			label: "the Second word",
			fillColor : "rgba(48, 164, 255, 0.2)",
			strokeColor : "rgba(48, 164, 255, 1)",
			pointColor : "rgba(48, 164, 255, 1)",
			pointStrokeColor : "#fff",
			pointHighlightFill : "#fff",
			pointHighlightStroke : "rgba(48, 164, 255, 1)",
			data : z
		}
	]

}



function createChart(x,y){
	
	window.myLine = new Chart(chart1).Line(lineChartData, {
	responsive: true,
	scaleLineColor: "rgba(0,0,0,.2)",
	scaleGridLineColor: "rgba(0,0,0,.05)",
	scaleFontColor: "#c5c7cc"
	});};

/*function piechart(Accuracy){
	
	/*ref.on("value", function(snapshot) {
		var Data=snapshot.val();
		var Accuracy=(Data.accuracy);//as a variable y in line chart 1 and variable in bie chart word1
		var Accuracy2=(Data.accuracy2);//as a variable y in line chart 2
		document.getElementById("easypiechart-blue").setAttribute("data-percent",String(Accuracy));
		
   });*/
	
	/*var show = msglist.getAttribute("data-percent");
	  console.log(show);
	msglist.removeAttribute("data-percent");
	//var show = msglist.getAttribute("data-percent");
	//console.log(show);
	msglist.setAttribute("data-percent", String(y));
	//var att = document.createAttribute("data-percent");        // Create a "href" attribute
    //att.value = "0"; 
	var show = msglist.getAttribute("data-percent");
	console.log(show);

};*/











v/*ar pieData = [
	{
		value: 50,
		color:"#30a5ff",
		highlight: "#62b9fb",
		label: "Blue"
	},
	{
		value: 50,
		color: "#ffb53e",
		highlight: "#fac878",
		label: "Orange"
	},
	{
		value: 100,
		color: "#1ebfae",
		highlight: "#3cdfce",
		label: "Teal"
	},
	{
		value: 220,
		color: "#f9243f",
		highlight: "#f6495f",
		label: "Red"
	}

];













	var barChartData = {
		labels : ["January","February","March","April","May","June","July"],
		datasets : [
			{
				fillColor : "rgba(220,220,220,0.5)",
				strokeColor : "rgba(220,220,220,0.8)",
				highlightFill: "rgba(220,220,220,0.75)",
				highlightStroke: "rgba(220,220,220,1)",
				data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
			},
			{
				fillColor : "rgba(48, 164, 255, 0.2)",
				strokeColor : "rgba(48, 164, 255, 0.8)",
				highlightFill : "rgba(48, 164, 255, 0.75)",
				highlightStroke : "rgba(48, 164, 255, 1)",
				data : [randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor(),randomScalingFactor()]
			}
		]

	}

			
	var doughnutData = [
				{
					value: 300,
					color:"#30a5ff",
					highlight: "#62b9fb",
					label: "Blue"
				},
				{
					value: 50,
					color: "#ffb53e",
					highlight: "#fac878",
					label: "Orange"
				},
				{
					value: 100,
					color: "#1ebfae",
					highlight: "#3cdfce",
					label: "Teal"
				},
				{
					value: 120,
					color: "#f9243f",
					highlight: "#f6495f",
					label: "Red"
				}

			];
			
	var radarData = {
	    labels: ["Eating", "Drinking", "Sleeping", "Designing", "Coding", "Cycling", "Running"],
	    datasets: [
	        {
	            label: "My First dataset",
	            fillColor: "rgba(220,220,220,0.2)",
	            strokeColor: "rgba(220,220,220,1)",
	            pointColor: "rgba(220,220,220,1)",
	            pointStrokeColor: "#fff",
	            pointHighlightFill: "#fff",
	            pointHighlightStroke: "rgba(220,220,220,1)",
	            data: [65, 59, 90, 81, 56, 55, 40]
	        },
	        {
	            label: "My Second dataset",
	            fillColor : "rgba(48, 164, 255, 0.2)",
	            strokeColor : "rgba(48, 164, 255, 0.8)",
	            pointColor : "rgba(48, 164, 255, 1)",
	            pointStrokeColor : "#fff",
	            pointHighlightFill : "#fff",
	            pointHighlightStroke : "rgba(48, 164, 255, 1)",
	            data: [28, 48, 40, 19, 96, 27, 100]
	        }
	    ]
	};
	
	var polarData = [
		    {
		    	value: 300,
		    	color: "#1ebfae",
		    	highlight: "#38cabe",
		    	label: "Teal"
		    },
		    {
		    	value: 140,
		    	color: "#ffb53e",
		    	highlight: "#fac878",
		    	label: "Orange"
		    },
		    {
		    	value: 220,
		    	color:"#30a5ff",
		    	highlight: "#62b9fb",
		    	label: "Blue"
		    },
		    {
		    	value: 250,
		    	color: "#f9243f",
		    	highlight: "#f6495f",
		    	label: "Red"
		    }
		
	];

*/