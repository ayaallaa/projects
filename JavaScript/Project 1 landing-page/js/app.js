/**
 * 
 * Manipulating the DOM exercise.
 * Exercise programmatically builds navigation,
 * scrolls to anchors from navigation,
 * and highlights section in viewport upon scrolling.
 * 
 * Dependencies: None
 * 
 * JS Version: ES2015/ES6
 * 
 * JS Standard: ESlint
 * 
*/
/*eslint-env es6*/
/**
 * Define Global Variables
 * 
*/
const navul = document.getElementById("navbar__list");
const fragment_nav = document.createDocumentFragment();
const sections =document.getElementsByClassName("landing__container");


// build the nav
for (const section of sections) {
    let a = document.createElement('a');
   let li = document.createElement('li');
    let text=section.firstElementChild.textContent;
    li.textContent = text  ;
    a.appendChild(li);
    fragment_nav.appendChild(a);

}
navul.appendChild(fragment_nav);
const navchilds=navul.childNodes;




// Build menu
let sec1top ;
let sec2top ;
let sec3top ;
let sec4top ;
function topsec (){
const sec1= document.getElementById("section1");
sec1top=sec1.offsetTop;
const sec2= document.getElementById("section2");
sec2top=sec2.offsetTop;
const sec3= document.getElementById("section3");
sec3top=sec3.offsetTop;
const sec4= document.getElementById("section4");
sec4top=sec4.offsetTop;
}
topsec ();
// Scroll to section on link click
navchilds[0].addEventListener('click', function () {
  window.scrollTo(
    {
  top: sec1top,
  behavior: 'smooth',
}
)});
navchilds[1].addEventListener('click', function () {
  window.scrollTo(
    {
  top: sec2top,
  behavior: 'smooth',
}
)});
navchilds[2].addEventListener('click', function () {
  window.scrollTo(
    {
  top: sec3top,
  behavior: 'smooth',
}
)});
navchilds[3].addEventListener('click', function () {
  window.scrollTo(
    {
  top: sec4top,
  behavior: 'smooth',
}
)});


// Set sections as active
let activeSec =navul;
 let textofnav ;
let activeNav ;
function activeSection (){

for (const section of sections) {
    let topSec = section.getBoundingClientRect().top; 
    //console.log(section);
    //console.log(topSec);
    if(topSec >= -50 && topSec <= 200)
        {
            activeSec=section;
            textofnav=activeSec.firstElementChild.textContent;
            
        }
    section.style.background= 'linear-gradient(0deg, rgba(255,255,255,.1) 0%, rgba(255,255,255,.2) 100%)';
    }
    
activeSec.style.background = '#2B60DE';
for (const navchild of navchilds){
      navchild.style.background = null;
    //navchild.style.background ="black";
if(textofnav==navchild.textContent)
    { activeNav=navchild;
      activeNav.style.background ="blue";
    
    }
    
}
}


//window.addEventListene ‘scroll’,
window.addEventListener('scroll', function () {
  topsec();
  activeSection();
});



//top button and scroll to top
const Btop = document.createElement('a');
const foot =document.getElementsByClassName('page__footer')[0] ;
Btop.innerHTML = '<button type="button">Top</button>';
foot.appendChild(Btop);
Btop.addEventListener('click', function () {
  //window.scrollTo(0, 0);
     window.scrollTo(
    {
  top: 0,
  behavior: 'smooth',
}
)});
