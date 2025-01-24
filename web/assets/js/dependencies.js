var footer = document.querySelector('footer')
var link = document.createElement('a')

// Add multiple classes correctly using the spread operator
link.classList.add('built-with', 'svelte-1ax1toq')
link.id = 'show_resource_monitor'
link.text = 'Resource Monitor'
link.onclick = function () {
  showPerfMonitor()
} // Use function reference instead of string

var linkImg = document.createElement('img')
linkImg.src = '/file=web/assets/img/monitor.svg'
linkImg.classList.add('svelte-1ax1toq')
link.appendChild(linkImg)
footer.appendChild(link)

var script = document.createElement('script')
script.src = '/file=web/assets/js/jquery-3.7.1.min.js'
document.body.appendChild(script)

var script = document.createElement('script')
script.src = '/file=web/assets/js/elegant-resource-monitor.js'
document.body.appendChild(script)

var script = document.createElement('script')
script.src = '/file=web/assets/js/socket.io.min.js'
document.body.appendChild(script)

var script = document.createElement('script')
script.src = '/file=web/assets/js/chart.js'
document.body.appendChild(script)

var fa = document.createElement('link')
fa.href = '/file=web/assets/css/material-icon.css'
fa.property = 'stylesheet'
fa.rel = 'stylesheet'
document.body.appendChild(fa)

var styles = document.createElement('link')
styles.href = '/file=web/assets/css/styles.css'
styles.property = 'stylesheet'
styles.rel = 'stylesheet'
document.body.appendChild(styles)
styles.onload = async function () {
  if (localStorage.getItem('lastClass') && localStorage.getItem('lastInactiveClass')) {
    var lastClass = JSON.parse(localStorage.getItem('lastClass'))
    var lastInactiveClass = JSON.parse(localStorage.getItem('lastInactiveClass'))
    addCSS(lastInactiveClass.key, lastInactiveClass.values[0])
    addCSS(lastClass.key, lastClass.values[0])
  }

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

  function addCSS(selector, styles) {
    var rule = getCSSRule(selector)

    for (var property in styles) {
      if (styles.hasOwnProperty(property)) {
        rule.style.setProperty(property, styles[property], 'important')
      }
    }
  }

  async function loadHtmlContent() {
    const response = await fetch('/file=web/templates/perf-monitor/perf-monitor.html')
    var resourceMonitorContent = document.getElementById('perf-monitor-container')
    resourceMonitorContent.innerHTML = await response.text()
    const chartButton = resourceMonitorContent.querySelector('#chart-button')
    const savedPosition = localStorage.getItem('perf-monitor-position') || 'bottom-right'

    if (chartButton) {
      // Set the savedPosition class on the #chart-button element
      chartButton.classList.add(savedPosition)
    }

    var script = document.createElement('script')
    script.src = '/file=web/assets/js/script.js'
    document.body.appendChild(script)

    var chart = document.createElement('script')
    chart.src = '/file=web/assets/js/chart-settings.js'
    document.body.appendChild(chart)
  }
  await loadHtmlContent()
}
