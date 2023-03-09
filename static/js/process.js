document.addEventListener("DOMContentLoaded", () => {
    let divImgs = document.getElementsByClassName("div-img")
    Array.from(divImgs).forEach((element, index, array) => {
        element.addEventListener("click", () => {
            let divClass = document.getElementById("divClass")
            let divInside = document.getElementById("divInside")
            divClass.style.height = "5%"
            divClass.style.width = "20%"
            divClass.style.marginLeft = "15%"
            divClass.style.justifyContent = "start"
            Array.from(divImgs).forEach((element, index, array) => {
                element.getElementsByTagName("img")[0].remove()
                element.getElementsByTagName("p")[0].style.fontSize = "2vmin"
            })
            console.log(divInside)
            divInside.style.opacity = "1"
        })
    });
})