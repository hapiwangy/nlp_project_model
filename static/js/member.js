document.addEventListener("DOMContentLoaded", () => {
    let btnRefer = document.getElementById("btnRefer")
    btnRefer.addEventListener("click", () => {
        console.log(btnRefer.textContent)
        let divRefer = document.getElementById("divRefer")
        divRefer.style.opacity = btnRefer.textContent === '參考資料' ? 1 : 0
        let divContent = document.getElementById("divContent")
        divContent.style.opacity = btnRefer.textContent === '參考資料' ? 0 : 1
        btnRefer.textContent = btnRefer.textContent === '參考資料' ? '返回' : '參考資料'
    })
})
