package main

import (
	"encoding/csv"
	"image/color"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type DataPoint struct {
	Date  time.Time
	Price float64
}

func loadData(filePath string) ([]DataPoint, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = ';'
	reader.FieldsPerRecord = -1

	if _, err := reader.Read(); err != nil {
		return nil, err
	}

	var dataPoints []DataPoint
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		timeStr := strings.Trim(record[0], "\"")
		priceStr := record[6]

		date, err := time.Parse("2006-01-02T15:04:05.000Z", timeStr)
		if err != nil {
			log.Printf("Błąd parsowania daty: %v", err)
			continue
		}

		price, err := strconv.ParseFloat(priceStr, 64)
		if err != nil {
			log.Printf("Błąd parsowania ceny: %v", err)
			continue
		}

		dataPoints = append(dataPoints, DataPoint{
			Date:  date,
			Price: price,
		})
	}

	return dataPoints, nil
}

func lpplModel(t, tc, m, omega, A, B, C, phi float64) float64 {
	dt := tc - t
	if dt <= 0 {
		return A
	}
	return A + B*math.Pow(dt, m)*(1+C*math.Cos(omega*math.Log(dt)+phi))
}

func lpplCost(params []float64, data []DataPoint, timeIndex []float64) float64 {
	tc, m, omega, A, B, C, phi := params[0], params[1], params[2], params[3], params[4], params[5], params[6]

	var sum float64
	for i, point := range data {
		t := timeIndex[i]
		predicted := lpplModel(t, tc, m, omega, A, B, C, phi)
		actual := math.Log(point.Price)
		sum += math.Pow(actual-predicted, 2)
	}
	return sum
}

func fitModel(data []DataPoint) ([]float64, error) {
	timeIndex := make([]float64, len(data))
	start := data[0].Date
	for i := range data {
		timeIndex[i] = data[i].Date.Sub(start).Hours() / 24
	}

	problem := optimize.Problem{
		Func: func(params []float64) float64 {
			return lpplCost(params, data, timeIndex)
		},
	}

	// Początkowe wartości parametrów
	initial := []float64{
		float64(len(data)) + 30, // tc
		0.7,                     // m (beta)
		8.0,                     // omega
		math.Log(data[0].Price), // A
		-1.0,                    // B
		0.1,                     // C
		0.0,                     // phi
	}

	result, err := optimize.Minimize(problem, initial, nil, nil)
	if err != nil {
		return nil, err
	}

	return result.X, nil
}

func plotResults(data []DataPoint, params []float64) error {
	p := plot.New()
	p.Title.Text = "Model LPPL - Bitcoin"
	p.X.Label.Text = "Dni od początku"
	p.Y.Label.Text = "Cena (USD)"

	// Dane rzeczywiste
	pts := make(plotter.XYs, len(data))
	start := data[0].Date
	for i := range data {
		pts[i].X = data[i].Date.Sub(start).Hours() / 24
		pts[i].Y = data[i].Price
	}

	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		return err
	}
	scatter.GlyphStyle.Color = color.RGBA{B: 255, A: 255}

	// Krzywa modelu
	tc := params[0]
	modelFunc := func(x float64) float64 {
		return math.Exp(lpplModel(x, tc, params[1], params[2], params[3], params[4], params[5], params[6]))
	}
	line := plotter.NewFunction(modelFunc)

	line.Color = color.RGBA{R: 255, A: 255}

	p.Add(scatter, line)
	p.Legend.Add("Dane", scatter)
	p.Legend.Add("Model LPPL", line)

	return p.Save(10*vg.Inch, 6*vg.Inch, "bitcoin_lppl.png")
}

func main() {
	data, err := loadData("Bitcoin_11.03.2025-10.04.2025_historical_data_coinmarketcap.csv")
	if err != nil {
		log.Fatal(err)
	}

	params, err := fitModel(data)
	if err != nil {
		log.Fatal(err)
	}

	log.Printf("Dopasowane parametry:")
	log.Printf("tc: %.2f dni", params[0])
	log.Printf("beta: %.4f", params[1])
	log.Printf("omega: %.4f", params[2])
	log.Printf("A: %.4f", params[3])
	log.Printf("B: %.4f", params[4])
	log.Printf("C: %.4f", params[5])
	log.Printf("phi: %.4f", params[6])

	if err := plotResults(data, params); err != nil {
		log.Fatal(err)
	}
}
