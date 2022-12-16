document.addEventListener("DOMContentLoaded", () => {
    const obj = JSON.parse(results);
    console.log(obj[0])
    data = document.getElementById("data")
    for (let i = 0; i < obj.length; i++) {
        data.append(obj[i] + '\n')
    }
})