function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


let USELESS_STUFF_HEIGHT = 69.79
let POINT_ELEMENT_HEIGHT = 54

let list = document.querySelector('[data-testid="virtuoso-item-list"]')
let list_scroller = document.querySelector('[data-testid="virtuoso-scroller"]')

//getting all the div elements in left sidebar
list_scroller.scrollTo(0, USELESS_STUFF_HEIGHT)
await delay(100)

//!in case needed later
// let previous_scroll_top = -1000
// let div_set = new Set()
// while(list_scroller.scrollTop - previous_scroll_top >= POINT_ELEMENT_HEIGHT){
//     let point_divs = list.querySelectorAll('div[data-index]')
//     for(point_div of point_divs){
//         div_set.add(point_div)
//     }
//     previous_scroll_top = list_scroller.scrollTop
//     list_scroller.scrollBy(0, POINT_ELEMENT_HEIGHT)
//     await delay(5)
// }

// //scrolling back to top
// list_scroller.scrollTo(0, USELESS_STUFF_HEIGHT)
// await delay(100)


// // for(let a = 2; a < div_set.size; a++){
// //     let div = list.querySelector('div[data-index="' + a + '"]')
// //     let buttons = div.querySelectorAll('button')
// //     buttons[1].click()
// //     await delay(100)
// // }

let count = 2;
let azimuth_array = []
while(true){
    //getting button and clicking it
    let div = list.querySelector('div[data-index="' + count + '"]')
    if(div == null){
        break
    }
    let buttons = div.querySelectorAll('button')
    buttons[1].click()

    await delay(50)
    let info_divs = document.querySelectorAll('div[class="_info-block_125w5_1 "]')
    let azimuth_p = info_divs[4].querySelectorAll('p[class="_item_19ga2_5"]')
    let azimuth_span = azimuth_p[1].querySelectorAll('span')
    
    let azimuth = azimuth_span[1].textContent
    azimuth = azimuth.substring(0, azimuth.length - 1)
    azimuth_array.push(azimuth)

    
    count++
}

console.log(azimuth_array)
