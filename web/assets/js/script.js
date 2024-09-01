dragElement(document.getElementById('chart-button-container'))

var wasDragged = false
function dragElement(elmnt) {
  var isDragging = false
  var pos1 = 0,
    pos2 = 0,
    pos3 = 0,
    pos4 = 0
  // otherwise, move the DIV from anywhere inside the DIV:
  elmnt.onmousedown = dragMouseDown
  function dragMouseDown(e) {
    e = e || window.event
    e.preventDefault()
    // get the mouse cursor position at startup:
    pos3 = e.clientX
    pos4 = e.clientY
    document.onmouseup = closeDragElement
    // call a function whenever the cursor moves:
    document.onmousemove = elementDrag
    elmnt.style.cursor = 'grabbing'
  }

  function elementDrag(e) {
    e = e || window.event
    e.preventDefault()
    // calculate the new cursor position:
    pos1 = pos3 - e.clientX
    pos2 = pos4 - e.clientY
    pos3 = e.clientX
    pos4 = e.clientY
    // set the element's new position:
    wasDragged = true
    isDragging = true
    elmnt.style.top = elmnt.offsetTop - pos2 + 'px'
    elmnt.style.left = elmnt.offsetLeft - pos1 + 'px'
  }

  function closeDragElement() {
    // stop moving when mouse button is released:
    document.onmouseup = null
    document.onmousemove = null
    elmnt.style.transition = 'transform .7s ease-in-out'
    elmnt.style.cursor = 'grab'
    setTimeout(() => {
      if (!isDragging) {
        elmnt.style.cursor = 'auto'
      }
    }, 1000)
    isDragging = false
    getNearestPosition()
  }
}

function moveButtonToCenter(duration = 300) {
  const { buttonHeight, buttonWidth } = getButtonSize()
  // Get button dimensions and viewport dimensions
  const widgetWidth = buttonWidth
  const widgetHeight = buttonHeight
  const viewportWidth = window.innerWidth
  const viewportHeight = window.innerHeight

  // Calculate center of the viewport
  const windowCenterX = viewportWidth / 2
  const windowCenterY = viewportHeight / 2

  // Calculate button center
  const buttonCenterX = widgetWidth / 2
  const buttonCenterY = widgetHeight / 2

  // Calculate the translation offsets needed to center the button
  const posx = windowCenterX - buttonCenterX
  const posy = windowCenterY - buttonCenterY

  goToPosition({ x: posx, y: posy })
}

// Call the function to move the button

// HELPER FUNCTIONS //

function getCSSRule(ruleName) {
  ruleName = ruleName.toLowerCase()
  var result = null
  var find = Array.prototype.find

  Array.prototype.find.call(document.styleSheets, (styleSheet) => {
    try {
      if (styleSheet.cssRules) {
        result = find.call(styleSheet.cssRules, (cssRule) => {
          return cssRule instanceof CSSStyleRule && cssRule.selectorText.toLowerCase() == ruleName
        })
      }
    } catch (e) {
      // Handle cross-origin or other access errors
      // console.info("Cannot access cssRules for stylesheet:", e);
    }
    return result != null
  })
  return result
}

window.barChart = async function () {
  checkForUpdates('active-chart', 'bar')
  await updateChartSize()
}

window.lineChart = async function () {
  checkForUpdates('active-chart', 'line')
  await updateChartSize()
}

window.smallChart = async function () {
  checkForUpdates('chart-size', 'small')
  await updateChartSize()
}

window.mediumChart = async function () {
  checkForUpdates('chart-size', 'medium')
  await updateChartSize()
}

window.largeChart = async function () {
  setTimeout(async () => {
    checkForUpdates('perf-monitor-position', 'center')
    checkForUpdates('chart-size', 'large')
    await updateChartSize()
  }, 50)
}

function moveToCenter() {
  if (localStorage.getItem('perf-monitor-position') === 'center') {
    moveButtonToCenter(150)
  }
}
function checkForUpdates(key, value) {
  var previous = localStorage.getItem(key)
  var updated = previous != value
  localStorage.setItem('hasUpdates', updated)
  localStorage.setItem(key, value)
}

function isWindowOutsideWorkingArea() {
  const { buttonHeight, buttonWidth } = getButtonSize()

  // Get display bounds
  const { displayBounds } = getDisplayAndWindowBounds()

  const widget = document.getElementById('chart-button-container')
  const rect = widget.getBoundingClientRect()
  const currentTop = rect.top + window.scrollY
  const currentLeft = rect.left + window.scrollX

  const windowLeft = currentLeft
  const windowTop = currentTop
  const windowRight = windowLeft + buttonWidth
  const windowBottom = windowTop + buttonHeight

  const displayLeft = 0
  const displayTop = 0
  const displayRight = displayLeft + displayBounds.width
  const displayBottom = displayTop + displayBounds.height
  let isOutside =
    windowLeft < displayLeft ||
    windowTop < displayTop ||
    windowRight > displayRight ||
    windowBottom > displayBottom

  if (isOutside) {
    console.log('The window is outside the working area.')
  } else {
    console.log('The window is within the working area.')
  }

  return isOutside
}

function getSizes() {
  const savedChart = localStorage.getItem('active-chart') ?? 'bar'
  var sizes = {}
  if (savedChart == 'bar') {
    sizes = {
      small: { height: '120', width: '150' },
      medium: { height: '300', width: '410' },
      large: { height: '450', width: '700' },
    }
  } else {
    sizes = {
      small: { height: '110', width: '160' },
      medium: { height: '245', width: '425' },
      large: { height: '380', width: '700' },
    }
  }
  return sizes
}

// SETTINGS MENU //
// POSITIONS BUTTONS
document.querySelectorAll('.position-clickable').forEach((button) => {
  button.addEventListener('click', async function () {
    const position = this.id
    wasDragged = false

    localStorage.setItem('perf-monitor-position', position)

    //the position we should be going to
    const pos = getCoordinates(false)
    if (pos) {
      goToPosition(pos)
    } else {
      console.error('Invalid position:', pos)
    }

    // Optionally hide the settings menu and adjust UI
    const settingsMenu = document.getElementById('settingsMenu')
    settingsMenu.classList.remove('show') // Hide the menu if visible

    document.querySelectorAll('.chart-row').forEach((row) => {
      row.classList.remove('no-drag')
      row.classList.add('drag')
    })
  })
})

// Show or hide the settings menu when the settings icon is clicked
document.getElementById('popupTrigger').addEventListener('click', function (event) {
  const settingsMenu = document.getElementById('settingsMenu')
  settingsMenu.classList.toggle('show') // Toggle the 'show' class for animation

  document.querySelectorAll('.chart-row').forEach((row) => {
    row.classList.add('no-drag')
    row.classList.remove('drag')
  })
  document.querySelectorAll('canvas').forEach((row) => {
    row.classList.add('no-drag')
    row.classList.remove('drag')
  })
  setTimeout(() => {
    const settingsMenuHr = document.getElementById('settings-hr')
    settingsMenuHr.classList.add('show') // Toggle the 'show' class for animation
  }, 300)

  event.stopPropagation()
})

// Hide the settings menu when clicking outside
window.addEventListener('click', function (e) {
  if (e.target.className.includes('settings')) {
    return
  }

  const settingsMenu = document.getElementById('settingsMenu')
  const trigger = document.getElementById('popupTrigger')
  if (!settingsMenu.contains(e.target) && e.target !== trigger) {
    settingsMenu.classList.remove('show') // Hide the menu if clicking outside
  }
  document.querySelectorAll('canvas').forEach((row) => {
    row.classList.remove('no-drag')
    row.classList.add('drag')
  })
  document.querySelectorAll('.chart-row').forEach((row) => {
    row.classList.remove('no-drag')
    row.classList.add('drag')
  })
})

// Calculate if the menu will overflow the bottom of the viewport
document.getElementById('popupTrigger').addEventListener('click', function () {
  const menu = document.getElementById('settingsMenu')
  const menuRect = menu.getBoundingClientRect()
  const buttonRect = this.getBoundingClientRect()
  const viewportHeight = window.innerHeight
  if (menu.offsetTop < 0) {
    menu.style.position = 'absolute'
    menu.style.top = `29px`
  }
  let topPosition = buttonRect.bottom
  if (topPosition + menuRect.height > viewportHeight) {
    // Calculate how much the menu overflows the viewport
    const overflowAmount = topPosition + menuRect.height - viewportHeight
    // Apply the calculated position
    menu.style.position = 'absolute' // Ensure the menu is positioned absolutely
    menu.style.top = `-${overflowAmount}px`
  }
})

function goToPosition(pos) {
  const widget = document.getElementById('chart-button-container')

  // Set transition for smooth animation
  // widget.style.transition = 'transform .7s ease-in-out'
  widget.style.transition = `top .4s ease, left .4s ease`

  const currentTop = +widget.style.top.replace('px', '')
  const currentLeft = +widget.style.left.replace('px', '')

  // Target position

  const offsetX = pos.x - currentLeft
  const offsetY = pos.y - currentTop

  // Set transition duration and easing
  widget.style.transition = `transform .7s ease-in-out`

  // Animate to the center
  widget.style.transform = `translate(${offsetX}px, ${offsetY}px)`
}
// MAIN METHODS //

function getDisplayAndWindowBounds() {
  const availWidth = window.screen.availWidth
  const availHeight = window.screen.availHeight

  // Assume work area starts at (0, 0)
  // Work area dimensions approximate available screen area minus some margins
  const workArea = {
    x: 0, // Typically starts at (0, 0)
    y: 0, // Typically starts at (0, 0)
    width: availWidth,
    height: availHeight,
  }

  const displayBounds = {
    width: window.screen.width,
    height: window.screen.height,
    availableWidth: window.screen.availWidth,
    availableHeight: window.screen.availHeight,
    workArea,
  }

  const windowBounds = {
    width: window.innerWidth,
    height: window.innerHeight,
  }

  return { displayBounds, windowBounds }
}

function getPositions() {
  const { buttonHeight, buttonWidth } = getButtonSize()

  // Get button dimensions and viewport dimensions
  const widgetWidth = buttonWidth
  const widgetHeight = buttonHeight
  const viewportWidth = window.innerWidth
  const viewportHeight = window.innerHeight

  // Calculate center of the viewport
  const windowCenterX = viewportWidth / 2
  const windowCenterY = viewportHeight / 2

  // Calculate button center
  const buttonCenterX = widgetWidth / 2
  const buttonCenterY = widgetHeight / 2

  // Calculate the translation offsets needed to center the button

  // Define positions based on work area
  const positions = {
    'bottom-right': {
      x: viewportWidth - widgetWidth - 10,
      y: viewportHeight - widgetHeight - 10,
    },
    'bottom-left': {
      x: 10,
      y: viewportHeight - widgetHeight - 10,
    },
    'bottom-center': {
      x: (viewportWidth - widgetWidth) / 2,
      y: viewportHeight - widgetHeight - 10,
    },
    'top-right': {
      x: viewportWidth - widgetWidth - 10,
      y: 10,
    },
    'top-left': { x: 10, y: 10 },
    'top-center': {
      x: (viewportWidth - widgetWidth) / 2,
      y: 10,
    },
    'left-center': {
      x: 10,
      y: windowCenterY - buttonCenterY,
    },
    'right-center': {
      x: viewportWidth - widgetWidth - 10,
      y: windowCenterY - buttonCenterY,
    },
    center: {
      x: (viewportWidth - widgetWidth) / 2,
      y: windowCenterY - buttonCenterY,
    },
  }
  return positions
}

function getCoordinates(isOutside) {
  var position = localStorage.getItem('perf-monitor-position')

  if (isOutside) {
    var outsidePosition = getNearestPosition()
    return outsidePosition
  }

  const positions = getPositions()
  const pos = positions[position]
  return pos
}

function getNearestPosition() {
  const { buttonHeight, buttonWidth } = getButtonSize()
  const widget = document.getElementById('chart-button-container')
  const viewportWidth = window.innerWidth
  const viewportHeight = window.innerHeight
  // Get display bounds
  const { displayBounds } = getDisplayAndWindowBounds()
  // Define positions based on work area
  const positions = getPositions()

  // Get current window position
  const currentX = $(widget).offset().left
  let currentY = $(widget).offset().top
  const windowCenter = {
    x: $(widget).offset().left,
    y: $(widget).offset().top,
  }
  const workAreaCenter = {
    x: viewportWidth / 2,
    y: viewportHeight / 2,
  }
  const distanceToCenter = {
    x: Math.abs(workAreaCenter.x - windowCenter.x),
    y: Math.abs(workAreaCenter.y - windowCenter.y),
  }
  var threshold = 100 // Define a threshold to determine proximity
  const size = localStorage.getItem('chart-size') ?? 'medium'
  switch (size) {
    case 'small':
      threshold = 250

      break
    case 'medium':
      threshold = 200
    default:
      threshold = 150

      break
  }

  var nearestPosition = ''

  if (distanceToCenter.x < threshold && distanceToCenter.y < threshold) {
    nearestPosition = 'center'
  } else {
    // Function to calculate distance
    function calculateDistance(x1, y1, x2, y2) {
      return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    }

    // Find the nearest position
    let minDistance = Infinity

    for (const [key, pos] of Object.entries(positions)) {
      // Adjust for edge cases
      const adjustedPosX = Math.max(
        displayBounds.workArea.x,
        Math.min(pos.x, displayBounds.width - buttonWidth),
      )
      const adjustedPosY = Math.max(
        displayBounds.workArea.y,
        Math.min(pos.y, displayBounds.height - buttonHeight),
      )

      const distance = calculateDistance(currentX, currentY, adjustedPosX, adjustedPosY)
      if (distance < minDistance) {
        minDistance = distance
        nearestPosition = key
      }
    }
  }

  // Output or use the nearest position
  console.log('Nearest position:', nearestPosition)
  // Set the position
  const pos = positions[nearestPosition]
  localStorage.setItem('perf-monitor-position', nearestPosition)

  return pos
}
function getButtonSize() {
  const size = localStorage.getItem('chart-size') ?? 'medium'
  const sizes = getSizes()
  const sizeStyles = sizes[size]
  var buttonHeight = +sizeStyles.height + 25
  const buttonWidth = +sizeStyles.width + 25
  var table = document.getElementById('item-table')
  var totalRowCount = table.rows.length // 5
  var tableHeight = totalRowCount * 30
  if (size == 'large') {
    buttonHeight = +buttonHeight + tableHeight
  }

  return { buttonHeight, buttonWidth }
}
async function updateChartSize() {
  var table = document.getElementById('item-table')
  table.style.display = 'none'

  const settingsMenu = document.getElementById('settingsMenu')
  settingsMenu.classList.remove('show') // Hide the menu if visible
  $('#chart-wrapper').fadeOut()
  document.querySelectorAll('.chart-row').forEach((row) => {
    row.classList.remove('no-drag')
    row.classList.add('drag')
  })

  document.querySelectorAll('canvas').forEach((row) => {
    row.classList.remove('no-drag')
    row.classList.add('drag')
  })

  const size = localStorage.getItem('chart-size') ?? 'medium'
  const chartContainer = document.getElementById('chart-container')
  const savedChart = localStorage.getItem('active-chart') ?? 'bar'

  chartContainer.classList.remove('small', 'medium', 'large', 'bar', 'line')
  chartContainer.classList.add(size)
  chartContainer.classList.add(savedChart)

  const { buttonHeight, buttonWidth } = getButtonSize()
  const chartButtonContainer = document.getElementById('chart-button-container')

  var actulaButtonHeight = 0
  var actualButtonWidth = 0

  var viewportHeight = +buttonHeight
  var viewportWidth = +buttonWidth

  $(chartButtonContainer).each(function () {
    this.style.setProperty('height', `${viewportHeight}px`, 'important')
    this.style.setProperty('width', `${viewportWidth}px`, 'important')
  })

  var position = localStorage.getItem('perf-monitor-position')
  if (position === 'center') {
    moveToCenter()
  } else {
    const isOutside = isWindowOutsideWorkingArea()
    const pos = getCoordinates(isOutside)
    if (pos && isOutside && wasDragged) {
      goToPosition(pos)
    } else if (pos && !wasDragged) {
      goToPosition(pos)
    } else {
      // do nothing
    }
  }

  var sizeClasses = ['small', 'medium', 'large']

  const chartButton = document.getElementById('chart-button')
  chartButton.classList.add(size)
  sizeClasses.forEach((prop) => {
    if (prop != size) {
      setTimeout(() => {
        chartButton.classList.remove(prop)
      }, 500)
    }
  })

  switch (size) {
    case 'small':
      actulaButtonHeight = viewportHeight * 0.83
      actualButtonWidth = viewportWidth * 0.83
      break
    case 'medium':
      actulaButtonHeight = viewportHeight * 0.93
      actualButtonWidth = viewportWidth * 0.93
      break
    default:
      actulaButtonHeight = viewportHeight * 0.96
      actualButtonWidth = viewportWidth * 0.96
      break
  }

  const bottom = `12.5px`
  const right = `12.5px`

  $(chartButton).each(function () {
    this.style.setProperty('bottom', bottom, 'important')
    this.style.setProperty('right', right, 'important')

    if (size === 'large') {
      this.style.setProperty('background-color', ` #000000d6`, 'important')
    } else {
      this.style.setProperty('background-color', ` #00000096`, 'important')
    }
  })
  var totalRowCount = table.rows.length // 5
  var tableHeight = totalRowCount * 30
  var workArea = actulaButtonHeight
  if (size == 'large') {
    workArea = viewportHeight - tableHeight
  }

  const hasUpdates = localStorage.getItem('hasUpdates') ?? 'false'
  if (hasUpdates === 'true') {
    if (savedChart == 'bar') {
      $(chartContainer).each(function () {
        this.style.setProperty('height', `${workArea * 0.95}px`, 'important')
      })

      initializeBarChart()
    } else {
      $(chartContainer).each(function () {
        this.style.setProperty('height', `${workArea * 0.87}px`, 'important')
      })
      setTimeout(() => {
        initializeLineChart()
      }, 500)
    }
  } else {
    $('#chart-wrapper').fadeIn()
  }

  localStorage.setItem('hasUpdates', 'false')

  const active = `#chart-button.top-left.active`
  var positionStyles = {
    bottom: bottom,
    right: right,
  }
  var lastClass = {
    key: active,
    values: [positionStyles],
  }
  var lastClassString = JSON.stringify(lastClass)
  localStorage.setItem('lastClass', lastClassString)
}

const pos = getCoordinates(false)
goToPosition(pos)
var appIsLoaded = false
var shouldShowPerfMonitor = false
if (JSON.parse(localStorage.getItem('shouldShowPerfMonitor')) ?? false) {
  setTimeout(() => {
    // showPerfMonitor()
  }, 1500)
}

window.showPerfMonitor = async function () {
  shouldShowPerfMonitor = !shouldShowPerfMonitor
  localStorage.setItem('shouldShowPerfMonitor', shouldShowPerfMonitor)
  const chartButton = document.getElementById('chart-button')
  const chartWrapper = document.getElementById('chart-wrapper')
  const chartButtonContainer = document.getElementById('chart-button-container')
  const resourceMonitorLink = document.getElementById('show_resource_monitor')

  if (shouldShowPerfMonitor === true) {
    $(chartButtonContainer).toggleClass('active')

    localStorage.setItem('hasUpdates', 'true')
    await updateChartSize()
    $(resourceMonitorLink).fadeOut(500)
    appIsLoaded = true
  } else {
    setTimeout(() => {
      $(chartButtonContainer).toggleClass('active')
      $(chartButtonContainer).each(function () {
        this.style.setProperty('height', `0px`, 'important')
      })
    }, 500)
    chartButton.classList.remove('small', 'medium', 'large')
    $(chartWrapper).fadeOut()
    $(resourceMonitorLink).fadeIn(500)
  }

  $(chartButton).toggleClass('active')
}

// when the close button is clicked
document.getElementById('close-button').addEventListener('click', function () {
  document.getElementById('settingsMenu').classList.remove('show') // Hide the menu
  document.querySelectorAll('.chart-row').forEach((row) => {
    row.classList.remove('no-drag')
    row.classList.add('drag')
  })
  showPerfMonitor()
})
