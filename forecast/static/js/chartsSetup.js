document.addEventListener('DOMContentLoaded', () => {
    const chartElement = document.getElementById('chart');
    if (!chartElement) {
        console.log('Canvas Element not found');
        return;
    }

    const ctx = chartElement.getContext('2d');
    const gradient = ctx.createLinearGradient(0, -10, 0, 100);
    gradient.addColorStop(0, 'rgba(250, 0, 0,1)');
    gradient.addColorStop(1, 'rgba(136,255,0,1)');

    const forecastItems = document.querySelectorAll('.forecast-item');  // Use querySelectorAll to get all forecast items

    const temps = [];
    const times = [];
    forecastItems.forEach(item => {
        const tempText = item.querySelector('.forecast-temp')?.textContent;
        const time = item.querySelector('.forecast-time')?.textContent;
        const humidity = item.querySelector('.forecast-humidity')?.textContent;

        // Ensure valid data and convert temperature to a number
        if (time && tempText && humidity) {
            const temp = parseFloat(tempText);  // Convert temp to number
            if (!isNaN(temp)) {
                times.push(time);
                temps.push(temp);
            }
        }
    });

    if (temps.length === 0 || times.length === 0) {
        console.error('Temp or time values are missing');
        return;
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [
                {
                    label: 'Celsius Degrees',
                    data: temps,
                    backgroundColor: gradient,
                    tension: 0.4,
                    borderWidth: 2,
                    PointRadius: 2,
                },
            ],
        },
        options: {  // Correct property name to lowercase 'options'
            plugins: {  // Correct property name to lowercase 'plugins'
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                },
                y: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                }
            },
            animation: {
                duration: 750,
            }
        }
    });
});
