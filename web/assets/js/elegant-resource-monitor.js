class ElegantResourceMonitor extends HTMLElement {
  constructor() {
    super()
    this.shadow = null
    this.currentPromptExecution = null
    this.onProgressUpdateBound = this.onProgressUpdate.bind(this)
    this.connected = false
  }

  render() {
    this.innerHTML = `
      <div id="chart-button-container" draggable="true">
        <div id="chart-button">
          <div class="chart-row">
            <div class="left-col">
              <i class="material-icons" id="popupTrigger">settings</i>
              <div id="settingsMenu" class="settings-menu">
                <div class="settings-row"><div class="settings-col">Settings</div></div>
                <hr id="settings-hr" class="settings-hr" />
                <div class="settings-row">
                <div class="settings-row">
                  <div class="settings-col">Layout:</div>
                  <div class="settings-col">
                    <a href="#" onclick="barChart()">1</a> |
                    <a href="#" onclick="lineChart()">2</a>
                  </div>
                </div>
                <div class="settings-row">
                  <div class="settings-col">Size:</div>
                  <div class="settings-col">
                    <a href="#" onclick="smallChart()">S</a> |
                    <a href="#" onclick="mediumChart()">M</a>
                  </div>
                </div>
                </div>
                <div class="settings-row">
                  <div class="settings-col">Position</div>
                  <div id="positionMenu" class="position-menu">
                    <button class="position-btn position-clickable" id="top-left"><i class="material-icons">north_west</i></button>
                    <button class="position-btn position-clickable" id="top-center"><i class="material-icons">north</i></button>
                    <button class="position-btn position-clickable" id="top-right"><i class="material-icons">north_east</i></button>
                    <button class="position-btn position-clickable" id="left-center"><i class="material-icons">west</i></button>
                    <button class="position-btn position-clickable" id="center" onclick="largeChart()"><i class="material-icons">radio_button_checked</i></button>
                    <button class="position-btn position-clickable" id="right-center"><i class="material-icons">east</i></button>
                    <button class="position-btn position-clickable" id="bottom-left"><i class="material-icons">south_west</i></button>
                    <button class="position-btn position-clickable" id="bottom-center"><i class="material-icons">south</i></button>
                    <button class="position-btn position-clickable" id="bottom-right"><i class="material-icons">south_east</i></button>
                  </div>
                </div>
              </div>
            </div>
            <div class="chart-col">
              <i class="material-icons" id="close-button">close</i>
            </div>
          </div>
          <div id="chart-wrapper">
           <div id="progress-bar-container">
              <div id="progress-bar"></div>
            </div>
            <div id="chart-container">
              <canvas id="usage-chart" style="width: 100%; height: 100%"></canvas>
            </div>
            <div id="custom-legend"></div>
            <div id="table-view">
            <table id="item-table">
                <tbody id="item-body">
                <!-- Table rows will be dynamically added here -->
                </tbody>
            </table>
            </div>
          </div>
        </div>
      </div>
    `
  }

  addEventListeners() {
    this.querySelector('#popupTrigger').addEventListener('click', () => {
      const settingsMenu = this.querySelector('#settingsMenu')
      settingsMenu.style.display = settingsMenu.style.display === 'block' ? 'none' : 'block'
    })

    this.querySelector('#close-button').addEventListener('click', () => {
      this.dispatchEvent(new CustomEvent('close', { bubbles: true, composed: true }))
    })
  }

  getSizes() {
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
  getButtonSize() {
    const size = localStorage.getItem('chart-size') ?? 'medium'
    const sizes = this.getSizes()
    const sizeStyles = sizes[size]
    var buttonHeight = +sizeStyles.height + 25
    const buttonWidth = +sizeStyles.width + 25
    var table = this.querySelector('#item-table')
    var totalRowCount = table.rows.length // 5
    var tableHeight = totalRowCount * 30
    if (size == 'large') {
      buttonHeight = +buttonHeight + tableHeight
    }

    return { buttonHeight, buttonWidth }
  }

  addListItem(itemContent) {
    const itemBody = this.querySelector('#item-body')
    const row = document.createElement('tr')
    const cell = document.createElement('td')
    cell.innerText = itemContent
    row.appendChild(cell)
    itemBody.appendChild(row)
  }

  get currentNodeId() {
    var _a, _b
    const prompt = this.currentPromptExecution
    const nodeId =
      ((_a = prompt === null || prompt === void 0 ? void 0 : prompt.errorDetails) === null ||
      _a === void 0
        ? void 0
        : _a.node_id) ||
      ((_b = prompt === null || prompt === void 0 ? void 0 : prompt.currentlyExecuting) === null ||
      _b === void 0
        ? void 0
        : _b.nodeId)
    return nodeId || null
  }

  adjustButtonHeight(e) {
    const chartButtonContainer = this.querySelector('#chart-button-container')

    const shouldShowPerfMonitor = JSON.parse(localStorage.getItem('shouldShowPerfMonitor')) ?? false
    if (!shouldShowPerfMonitor || chartButtonContainer.clientHeight == 0) return

    const { buttonHeight, buttonWidth } = this.getButtonSize()
    const chartContainer = this.querySelector('#chart-container')
    const savedChart = localStorage.getItem('active-chart') ?? 'bar'
    const size = localStorage.getItem('chart-size') ?? 'medium'

    const height = this.querySelector('#chart-container').style.height.replace('px', '')
    let bar = this.querySelector('#progress-bar-container')

    var actulaButtonHeight = 0
    var actualButtonWidth = 0

    var viewportHeight = +buttonHeight
    var viewportWidth = +buttonWidth

    $(chartButtonContainer).each(function () {
      this.style.setProperty('height', `${viewportHeight}px`, 'important')
      this.style.setProperty('width', `${viewportWidth}px`, 'important')
    })
    var table = this.querySelector('#item-table')
    var totalRowCount = table.rows.length // 5
    var tableHeight = totalRowCount * 35
    var workArea = actulaButtonHeight
    if (size == 'large') {
      workArea = viewportHeight - tableHeight
    }

    const prompt = e.detail.prompt
    if (prompt && prompt.currentlyExecuting) {
      bar.style.display = 'block'
      workArea = workArea - 30
    } else {
      bar.style.display = 'none'
    }

    if (savedChart == 'bar') {
      $(chartContainer).each(function () {
        this.style.setProperty('height', `${workArea * 0.95}px`, 'important')
      })
    } else {
      $(chartContainer).each(function () {
        this.style.setProperty('height', `${workArea * 0.87}px`, 'important')
      })
    }
  }

  onProgressUpdate(e) {
    // this.adjustButtonHeight(e)
    var _a, _b, _c, _d
    if (!this.connected) return
    const prompt = e.detail.prompt

    this.currentPromptExecution = prompt

    // Default progress to 0 if no totalNodes
    let progressPercentage = 0

    if (prompt === null || prompt === void 0 ? void 0 : prompt.errorDetails) {
      let progressText = `${
        (_a = prompt.errorDetails) === null || _a === void 0 ? void 0 : _a.exception_type
      } ${((_b = prompt.errorDetails) === null || _b === void 0 ? void 0 : _b.node_id) || ''} ${
        ((_c = prompt.errorDetails) === null || _c === void 0 ? void 0 : _c.node_type) || ''
      }`
      console.log(progressText)
      // Set the progress bar to 0% or some error state if needed
      this.querySelector('#progress-bar').style.width = '0%'
      return
    }
    if (prompt === null || prompt === void 0 ? void 0 : prompt.currentlyExecuting) {
      const current = prompt === null || prompt === void 0 ? void 0 : prompt.currentlyExecuting

      let progressText = `(${e.detail.queue}) `
      if (!prompt.totalNodes) {
        progressText += `??%`
      } else {
        progressPercentage = (prompt.executedNodeIds.length / prompt.totalNodes) * 100
        progressText += `${Math.round(progressPercentage)}%`
      }

      let nodeLabel = (_d = current.nodeLabel) === null || _d === void 0 ? void 0 : _d.trim()
      const monitor = document.querySelector('elegant-resource-monitor')
      // monitor.addListItem(nodeLabel)

      let stepsLabel = ''
      if (current.step != null && current.maxSteps) {
        const percent = (current.step / current.maxSteps) * 100
        if (current.pass > 1 || current.maxPasses != null) {
          stepsLabel += `#${current.pass}`
          if (current.maxPasses && current.maxPasses > 0) {
            stepsLabel += `/${current.maxPasses}`
          }
          stepsLabel += ` - `
        }
        stepsLabel += `${Math.round(percent)}%`
      }

      if (nodeLabel || stepsLabel) {
        progressText += ` - ${nodeLabel || '???'}${stepsLabel ? ` (${stepsLabel})` : ''}`
      }
      console.log(progressText)
    } else {
      if (e === null || e === void 0 ? void 0 : e.detail.queue) {
        console.log(`(${e.detail.queue}) Running... in another tab`)
      } else {
        console.log('Idle')
      }
    }

    // Update the progress bar width
    this.querySelector('#progress-bar').style.width = `${Math.round(progressPercentage)}%`
  }

  connectedCallback() {
    if (!this.connected) {
      console.log('Adding event listener to MONITOR_SERVICE')
      // MONITOR_SERVICE.addEventListener(
      //   'elegant-resource-monitor-update',
      //   this.onProgressUpdateBound,
      // )

      this.render()
      this.addEventListeners()

      this.connected = true
    }
  }

  disconnectedCallback() {
    this.connected = false
    // MONITOR_SERVICE.removeEventListener(
    //   'elegant-resource-monitor-update',
    //   this.onProgressUpdateBound,
    // )
  }
}

// Register the custom element
customElements.define('elegant-resource-monitor', ElegantResourceMonitor)
