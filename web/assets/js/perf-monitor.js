window.barChart = function () {
  checkForUpdates("active-chart", "bar");
  updateChartSize();
};

window.lineChart = function () {
  checkForUpdates("active-chart", "line");
  updateChartSize();
};

window.smallChart = function () {
  checkForUpdates("chart-size", "small");
  updateChartSize();
};

window.mediumChart = function () {
  checkForUpdates("chart-size", "medium");
  updateChartSize();
};

window.largeChart = function () {
  checkForUpdates("perf-monitor-position", "center");
  checkForUpdates("chart-size", "large");
  updateChartSize();
};

function checkForUpdates(key, value) {
  var previous = localStorage.getItem(key);
  var updated = previous != value;
  localStorage.setItem("hasUpdates", updated);
  localStorage.setItem(key, value);
}

const { hostname } = window.location; // Gets the host without port
const baseUrl = `http://${hostname}:5000`; // Append the port 5000
const apiUrl = `${baseUrl}/gpu_usage/`;
const chartContainer = document.getElementById("chart-container");
const chartWrapper = document.getElementById("chart-wrapper");

var styles = document.createElement("link");
styles.href =
  "extensions/ComfyUI-Elegant-Resource-Monitor/assets/css/styles.css";
styles.property = "stylesheet";
styles.rel = "stylesheet";
document.head.appendChild(styles);

// Define your color palette
const colorPalette = [
  "rgb(240, 193, 90, 0.2)",
  "rgb(240, 142, 219, 0.2)",
  "rgb(24, 90, 219, 0.2)",
  "rgb(127, 161, 195, 0.2)",
  "rgb(128, 239, 145, 0.2)",
  "rgb(245, 245, 245, 0.2)",
  "rgb(240, 142, 219, 0.2)",
  "rgb(159, 238, 209, 0.2)",
];

const borderColors = [
  "rgb(240, 193, 90)",
  "rgb(240, 142, 219)",
  "rgb(24, 90, 219)",
  "rgb(127, 161, 195)",
  "rgb(128, 239, 145)",
  "rgb(245, 245, 245)",
  "rgb(240, 142, 219)",
  "rgb(159, 238, 209)",
];

// Custom plugin to draw fixed labels in the middle of the chart area
const fixedLabelPlugin = {
  id: "fixedLabelPlugin",
  afterDatasetsDraw(chart) {
    const { ctx, scales, data } = chart;
    ctx.save();

    const centerX = scales.x.left + scales.x.width / 2;
    const labelPositions = [];
    data.datasets[0].data.forEach((value, index) => {
      const yPos = chart.getDatasetMeta(0).data[index].y;

      // Store yPos for positioning labels
      labelPositions.push({
        x: centerX,
        y: yPos,
        value: `${value.toFixed(2)}` + `${index == 5 ? "°" : "%"}`,
      });
    });

    ctx.font = "8px Arial";
    ctx.fillStyle = "#FFFFFF";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    labelPositions.forEach((label) => {
      ctx.fillText(label.value, label.x, label.y);
    });

    ctx.restore();
  },
};

let currentChart = null; // Track the current chart instance
const MAX_DATA_POINTS = 50; // Number of data points to keep
function getSizes() {
  const size = localStorage.getItem("chart-size") ?? "small";
  const savedChart = localStorage.getItem("active-chart") ?? "bar";
  var sizes = {};
  if (savedChart == "bar") {
    sizes = {
      small: { height: "130", width: "180" },
      medium: { height: "220", width: "340" },
      large: { height: "440", width: "750" },
    };
  } else {
    sizes = {
      small: { height: "140", width: "200" },
      medium: { height: "255", width: "425" },
      large: { height: "450", width: "800" },
    };
  }
  return sizes;
}

function updateButtonPosition() {
  const size = localStorage.getItem("chart-size") ?? "small";
  const sizes = getSizes();
  const sizeStyles = sizes[size];
  const buttonHeight = sizeStyles.height;
  const buttonWidth = sizeStyles.width;
  const viewportHeight = window.innerHeight;
  const viewportWidth = window.innerWidth;
  setButtonPosition(buttonHeight, buttonWidth, viewportHeight, viewportWidth);
}

function updateChartSize() {
  const settingsMenu = document.getElementById("settingsMenu");
  settingsMenu.classList.remove("show"); // Hide the menu if visible
  const chartButton = document.getElementById("chart-button");
  const size = localStorage.getItem("chart-size") ?? "small";
  const savedChart = localStorage.getItem("active-chart") ?? "bar";
  const chartContainer = document.getElementById("chart-container");
  const sizes = getSizes();
  chartContainer.classList.remove("small", "medium", "large", "bar", "line");
  chartContainer.classList.add(size);
  chartContainer.classList.add(savedChart);

  const sizeStyles = sizes[size];
  const buttonHeight = sizeStyles.height;
  const buttonWidth = sizeStyles.width;
  $(chartButton).each(function () {
    this.style.setProperty("height", `${buttonHeight}px`, "important");
    this.style.setProperty("width", `${buttonWidth}px`, "important");
    if (size === "large") {
      this.style.setProperty("background-color", ` #000000d6`, "important");
    } else {
      this.style.setProperty("background-color", ` #00000096`, "important");
    }
  });

  updateButtonPosition();
  const hasUpdates = localStorage.getItem("hasUpdates") ?? "false";

  if (hasUpdates === "true") {
    if (savedChart == "bar") {
      initializeBarChart();
    } else {
      initializeLineChart();
    }
  }
}

function setButtonPosition(
  buttonHeight,
  buttonWidth,
  viewportHeight,
  viewportWidth
) {
  const positions = {
    "bottom-right": { bottom: "10px", right: "10px" },
    "bottom-left": {
      bottom: "10px",
      right: `${viewportWidth - buttonWidth - 10}px`,
    },
    "bottom-center": {
      bottom: "10px",
      right: `${(viewportWidth - buttonWidth) / 2}px`,
    },
    "top-right": {
      bottom: `${viewportHeight - buttonHeight - 10}px`,
      right: "10px",
    },
    "top-left": {
      bottom: `${viewportHeight - buttonHeight - 10}px`,
      right: `${viewportWidth - buttonWidth - 10}px`,
    },
    "top-center": {
      bottom: `${viewportHeight - buttonHeight - 10}px`,
      right: `${(viewportWidth - buttonWidth) / 2}px`,
    },
    "left-center": {
      bottom: `${(viewportHeight - buttonHeight) / 2}px`,
      right: `${viewportWidth - buttonWidth - 10}px`,
    },
    "right-center": {
      bottom: `${(viewportHeight - buttonHeight) / 2}px`,
      right: "10px",
    },
    center: {
      bottom: `${(viewportHeight - buttonHeight) / 2}px`,
      right: `${(viewportWidth - buttonWidth) / 2}px`,
    },
  };
  // Get the saved position
  const savedPosition =
    localStorage.getItem("perf-monitor-position") || "bottom-right";

  const chartButton = document.getElementById("chart-button");
  const existingClasses = [
    "bottom-right",
    "bottom-left",
    "bottom-center",
    "top-right",
    "top-left",
    "top-center",
    "left-center",
    "right-center",
    "center",
  ];
  existingClasses.forEach((cls) => {
    chartButton.classList.remove(cls);
  });
  chartButton.classList.add(savedPosition);

  const active = `#chart-button.${savedPosition}.active`;
  const positionStyles = positions[savedPosition];

  var lastClass = {
    key: active,
    values: [
      {
        bottom: positionStyles.bottom,
        right: positionStyles.right,
      },
    ],
  };
  var lastClassString = JSON.stringify(lastClass);
  localStorage.setItem("lastClass", lastClassString);

  updateCSS(active, positionStyles);

  const inactive = `#chart-button.${savedPosition}`;
  const inactiveStyles = {
    buttonHeight: buttonHeight,
    buttonWidth: buttonWidth,
    viewportHeight: viewportHeight,
    viewportWidth: viewportWidth,
  };
  updateinActiveCSS(inactive, inactiveStyles, savedPosition);
}
function updateinActiveCSS(selector, styles, key) {
  var button = getCSSRule(selector);
  var style = {
    bottom: "auto",
    right: "auto",
  };

  var buttonHeight = +styles.buttonHeight;
  var buttonWidth = +styles.buttonWidth;
  var viewportHeight = +styles.viewportHeight;
  var viewportWidth = +styles.viewportWidth;

  switch (key) {
    case "bottom-right":
      style.bottom = "10px";
      style.right = `-${buttonWidth + 210}px`;
      break;
    case "bottom-left":
      style.bottom = "10px";
      style.right = `calc(100vw +  ${buttonWidth + 210}px)`;
      break;

    case "bottom-center":
      style.bottom = `-${buttonHeight + 210}px`;
      style.right = `${(viewportWidth - buttonWidth) / 2}px`;
      break;

    case "top-right":
      style.bottom = `${viewportHeight - buttonHeight - 10}px`;
      style.right = `-${buttonWidth + 210}px`;
      break;

    case "top-left":
      style.bottom = `${viewportHeight - buttonHeight - 10}px`;
      style.right = `calc(100vw +  ${buttonWidth + 210}px)`;
      break;

    case "top-center":
      style.bottom = `calc(100vh + 30px + ${buttonHeight + 210}px)`;
      style.right = `${(viewportWidth - buttonWidth) / 2}px`;
      break;

    case "left-center":
      style.bottom = `${(viewportHeight - buttonHeight) / 2}px`;
      style.right = `calc(100vw + ${buttonWidth + 210}px)`;
      break;

    case "right-center":
      style.bottom = `${(viewportHeight - buttonHeight) / 2}px`;
      style.right = `-${buttonWidth + 210}px`;
      break;

    case "center":
      style.bottom = `calc(0vh - 30px - ${buttonHeight + 210}px)`;
      style.right = `${(viewportWidth - buttonWidth) / 2}px`;
      break;

    default:
      break;
  }
  button.style.setProperty("bottom", style.bottom, "important");
  button.style.setProperty("right", style.right, "important");
  var lastClass = {
    key: selector,
    values: [
      {
        bottom: style.bottom,
        right: style.right,
      },
    ],
  };
  var lastClassString = JSON.stringify(lastClass);
  localStorage.setItem("lastInactiveClass", lastClassString);
}

function updateCSS(selector, styles) {
  var button = getCSSRule(selector);
  button.style.setProperty("bottom", styles.bottom, "important");
  button.style.setProperty("right", styles.right, "important");
}

function getCSSRule(ruleName) {
  ruleName = ruleName.toLowerCase();
  var result = null;
  var find = Array.prototype.find;

  Array.prototype.find.call(document.styleSheets, (styleSheet) => {
    try {
      if (styleSheet.cssRules) {
        result = find.call(styleSheet.cssRules, (cssRule) => {
          return (
            cssRule instanceof CSSStyleRule &&
            cssRule.selectorText.toLowerCase() == ruleName
          );
        });
      }
    } catch (e) {
      // Handle cross-origin or other access errors
      // console.info("Cannot access cssRules for stylesheet:", e);
    }
    return result != null;
  });
  return result;
}

let intervalId; // Variable to store the interval ID
// Function to start the interval
function startInterval() {
  intervalId = setInterval(updateUsage, 500);
}
// Function to stop the interval
function stopInterval() {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null; // Optional: Reset intervalId to indicate no active interval
  }
}

// Initialize the bar chart
function initializeBarChart() {
  localStorage.setItem("active-chart", "bar");
  const chartContainer = document.getElementById("chart-container");
  const existingCanvas = document.getElementById("usage-chart");
  const chartWrapper = document.getElementById("chart-wrapper");
  if (existingCanvas) {
    chartContainer.removeChild(existingCanvas);
  }

  // Create a new canvas element
  const newCanvas = document.createElement("canvas");
  newCanvas.id = "usage-chart";
  newCanvas.classList.add("bar"); // Add the class directly to the canvas element
  chartContainer.appendChild(newCanvas);

  const ctx = newCanvas.getContext("2d");
  $(chartWrapper).hide();

  currentChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["CPU", "RAM", "GPU", "VRAM", "HDD", "TEMP"],
      datasets: [
        {
          label: "Usage",
          data: [0, 0, 0, 0, 0],
          barPercentage: 0.8, // Adjust space occupied by bars
          categoryPercentage: 1, // Adjust space between bars
          backgroundColor: function (context) {
            const value = context.dataset.data[context.dataIndex];
            return value > 90 ? "#D9534F" : colorPalette[context.dataIndex];
          },
          borderColor: function (context) {
            const value = context.dataset.data[context.dataIndex];
            return value > 90 ? "#D9534F" : borderColors[context.dataIndex];
          },
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      indexAxis: "y", // Horizontal bars
      scales: {
        x: {
          grid: {
            display: false, // Hide all grid lines
          },
          border: {
            display: false, // Hide all grid lines
          },
          beginAtZero: true,
          max: 100,
          ticks: {
            color: "#ffffff",
            font: {
              size: 7,
              weight: 600,
            },
            align: "center",
            callback: function (value, index, ticks) {
              return value + "%";
            },
          },
        },
        y: {
          grid: {
            display: false,
          },
          border: {
            color: "#ffffff30",
            width: 1, // Width of the axis border
          },
          ticks: {
            color: "#FFFFFF",
            crossAlign: "far",
            font: {
              weight: 600,
            },
            // Specify the maximum number of ticks to show
            maxTicksLimit: 10,
            // Control the step size between ticks
            stepSize: 1,
            // Optional: Set font size and other style properties
            font: {
              size: 7,
            },
          },
        },
      },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          enabled: false,
        },
      },
      responsive: true,
      maintainAspectRatio: false,
    },
    plugins: [fixedLabelPlugin], // Register the custom plugins
  });

  currentChart.options.animation = true;
  const legendContainer = document.getElementById("custom-legend");
  legendContainer.innerHTML = "";

  document.getElementById("settingsMenu").classList.remove("show"); // Hide the menu
  window.addEventListener("resize", () => {
    currentChart.resize();
  });
  $(chartWrapper).fadeIn(300);
}

// Initialize the line chart
function initializeLineChart() {
  localStorage.setItem("active-chart", "line");
  const existingCanvas = document.getElementById("usage-chart");
  const chartContainer = document.getElementById("chart-container");
  const chartWrapper = document.getElementById("chart-wrapper");
  if (existingCanvas) {
    chartContainer.removeChild(existingCanvas);
  }

  // Create a new canvas element
  const newCanvas = document.createElement("canvas");
  newCanvas.id = "usage-chart";
  newCanvas.classList.add("line"); // Add the class directly to the canvas element
  chartContainer.appendChild(newCanvas);
  $(chartWrapper).hide();

  const ctx = newCanvas.getContext("2d");

  // ctx.width = "225px";
  // ctx.height = "125px";
  currentChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "CPU",
          data: [],
          borderColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return borderColors[datasetIndex % borderColors.length];
          },
          borderWidth: 1.5,
          backgroundColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return colorPalette[datasetIndex % borderColors.length];
          },
          fill: false,
          tension: 0.1,
        },
        {
          label: "RAM",
          data: [],
          borderColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return borderColors[datasetIndex % borderColors.length];
          },
          borderWidth: 1.5,
          backgroundColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return colorPalette[datasetIndex % borderColors.length];
          },
          fill: false,
          tension: 0.1,
        },
        {
          label: "GPU",
          data: [],
          borderColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return borderColors[datasetIndex % borderColors.length];
          },
          borderWidth: 1.5,
          backgroundColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return colorPalette[datasetIndex % borderColors.length];
          },
          fill: false,
          tension: 0.1,
        },
        {
          label: "VRAM",
          data: [],
          borderColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return borderColors[datasetIndex % borderColors.length];
          },
          borderWidth: 1.5,
          backgroundColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return colorPalette[datasetIndex % borderColors.length];
          },
          fill: false,
          tension: 0.1,
        },
        {
          label: "HDD",
          data: [],
          borderColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return borderColors[datasetIndex % borderColors.length];
          },
          borderWidth: 1.5,
          backgroundColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return colorPalette[datasetIndex % borderColors.length];
          },
          fill: false,
          tension: 0.1,
        },
        {
          label: "TEMP",
          data: [],
          borderColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return borderColors[datasetIndex % borderColors.length];
          },
          borderWidth: 1.5,
          backgroundColor: function (context) {
            const dataset = context.dataset;
            const datasetIndex = context.datasetIndex;
            const shouldUseRed = dataset.data.some((value) => value > 90);

            if (shouldUseRed) {
              return "#D9534F"; // Return red color if any value exceeds 90
            }
            return colorPalette[datasetIndex % borderColors.length];
          },
          fill: false,
          tension: 0.1,
        },
      ],
    },
    options: {
      animation: {
        enabled: false,
        tension: {
          duration: 1000,
          easing: "linear",
          from: 1,
          to: 0,
          loop: true,
        },
      },
      elements: {
        point: {
          radius: 0,
        },
      },
      scales: {
        x: {
          ticks: {
            display: false,
          },
        },
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            color: "#FFFFFF",
            crossAlign: "far",
            padding: 0,
            font: {
              weight: 600,
              size: 7,
            },
            callback: function (value, index, ticks) {
              return value + "%";
            },
          },
        },
      },
      responsive: true,
      plugins: {
        legend: {
          display: true,
          labels: {
            generateLabels: false,
          },
        },
        title: {
          display: false,
        },
      },
    },
  });

  currentChart.options.animation = false;
  generateCustomLegend();
  document.getElementById("settingsMenu").classList.remove("show"); // Hide the menu

  window.addEventListener("resize", () => {
    currentChart.resize();
  });
  $(chartWrapper).fadeIn(300);
}

function generateCustomLegend() {
  const legendContainer = document.getElementById("custom-legend");
  legendContainer.innerHTML = "";

  currentChart.data.datasets.forEach((dataset, index) => {
    const legendItem = document.createElement("div");
    legendItem.className = "custom-legend-item";

    // Create text element
    const legendText = document.createElement("span");
    legendText.className = "custom-legend-text";
    legendText.textContent = dataset.label;
    const shouldUseRed = dataset.data.some((value) => value > 90);
    legendText.style.color = shouldUseRed
      ? "#D9534F"
      : `${borderColors[index]}`;

    legendText.style.fontWeight = shouldUseRed ? "700" : `400`;
    legendText.style.fontSize = "10px";

    legendItem.appendChild(legendText);
    legendContainer.appendChild(legendItem);
  });
}
async function updateUsage() {
  try {
    const response = await fetch(apiUrl);
    const data = await response.json();
    const timestamp = new Date();

    if (currentChart) {
      if (currentChart.config.type === "bar") {
        // Update data for bar chart
        currentChart.data.datasets[0].data = [
          data.cpu,
          data.ram,
          data.gpu,
          data.vram,
          data.hdd,
          data.temp,
        ];
      } else if (currentChart.config.type === "line") {
        // Update data for line chart
        currentChart.data.labels.push(timestamp);
        currentChart.data.datasets[0].data.push(data.cpu);
        currentChart.data.datasets[1].data.push(data.ram);
        currentChart.data.datasets[2].data.push(data.gpu);
        currentChart.data.datasets[3].data.push(data.vram);
        currentChart.data.datasets[4].data.push(data.hdd);
        currentChart.data.datasets[5].data.push(data.temp);

        // Prune old data if the number of points exceeds the limit
        if (currentChart.data.labels.length > MAX_DATA_POINTS) {
          currentChart.data.labels.shift(); // Remove the oldest label
          currentChart.data.datasets.forEach((dataset) => dataset.data.shift()); // Remove the oldest data points
        }
        generateCustomLegend();
      }

      // Update the chart with new data
      currentChart.update();
    }
  } catch (error) {
    console.error("Failed to fetch usage data.", error);
  }
}

// Show or hide the settings menu when the settings icon is clicked
document
  .getElementById("popupTrigger")
  .addEventListener("click", function (event) {
    const settingsMenu = document.getElementById("settingsMenu");
    settingsMenu.classList.toggle("show"); // Toggle the 'show' class for animation

    setTimeout(() => {
      const settingsMenuHr = document.getElementById("settings-hr");
      settingsMenuHr.classList.add("show"); // Toggle the 'show' class for animation
    }, 300);

    event.stopPropagation();
  });

// Hide the settings menu when the close button is clicked
document.getElementById("close-button").addEventListener("click", function () {
  document.getElementById("settingsMenu").classList.remove("show"); // Hide the menu
  showPerfMonitor();
});

// Hide the settings menu when clicking outside
window.addEventListener("click", function (e) {
  const settingsMenu = document.getElementById("settingsMenu");
  const trigger = document.getElementById("popupTrigger");
  if (!settingsMenu.contains(e.target) && e.target !== trigger) {
    settingsMenu.classList.remove("show"); // Hide the menu if clicking outside
  }
});

document.querySelectorAll(".position-clickable").forEach((button) => {
  button.addEventListener("click", function () {
    const position = this.id;

    localStorage.setItem("perf-monitor-position", position);
    updateButtonPosition();

    const settingsMenu = document.getElementById("settingsMenu");
    settingsMenu.classList.remove("show"); // Hide the menu if visible
  });
});

const perfMonitordisplayed = JSON.parse(
  localStorage.getItem("shouldShowPerfMonitor")
);

if (perfMonitordisplayed == true) {
  updateButtonPosition();

  setTimeout(() => {
    showPerfMonitor();
  }, 1000);
}

var shouldShowPerfMonitor = false;

window.showPerfMonitor = function () {
  // Set the initial position based on localStorage

  updateChartSize();
  shouldShowPerfMonitor = !shouldShowPerfMonitor;
  localStorage.setItem("shouldShowPerfMonitor", shouldShowPerfMonitor);
  const chartButton = document.getElementById("chart-button");
  const show_resource_monitor = document.getElementById(
    "show_resource_monitor"
  );

  if (shouldShowPerfMonitor === true) {
    const savedChart = localStorage.getItem("active-chart") ?? "bar";

    setTimeout(() => {
      if (savedChart == "bar") {
        initializeBarChart();
      } else {
        initializeLineChart();
      }
    }, 100);

    startInterval();
    $(show_resource_monitor).fadeOut();
  } else {
    setTimeout(() => {
      stopInterval();
    }, 500);
    $(chartButton).each(function () {
      this.style.setProperty("height", `${0}px`, "important");
    });
    $(chartWrapper).hide();
    $(show_resource_monitor).fadeIn();
  }
  $(chartButton).toggleClass("active");
};

document.getElementById("popupTrigger").addEventListener("click", function () {
  const menu = document.getElementById("settingsMenu");
  const menuRect = menu.getBoundingClientRect();
  const buttonRect = this.getBoundingClientRect();
  const viewportHeight = window.innerHeight;

  if (menu.offsetTop < 0) {
    menu.style.position = "absolute"; // Ensure the menu is positioned absolutely
    menu.style.top = `29px`;
  }

  // Default position: directly below the button
  let topPosition = buttonRect.bottom;

  // Calculate if the menu will overflow the bottom of the viewport
  if (topPosition + menuRect.height > viewportHeight) {
    // Calculate how much the menu overflows the viewport
    const overflowAmount = topPosition + menuRect.height - viewportHeight;
    // Apply the calculated position
    menu.style.position = "absolute"; // Ensure the menu is positioned absolutely
    menu.style.top = `-${overflowAmount}px`;
  }
});
