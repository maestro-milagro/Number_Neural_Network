package main

import (
	"flag"
	"fmt"
)

func main() {
	net := NewNetwork(784, 200, 10, 0.1)

	mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")

	flag.Parse()

	switch *mnist {
	case "train":
		err := net.mnistTrain()
		if err != nil {
			fmt.Printf("Error while train network: %v", err)
		}
		err = net.save()
		if err != nil {
			fmt.Printf("Error while saving results: %v", err)
		}
	case "predict":
		err := net.load()
		if err != nil {
			fmt.Printf("Error while loading weights: %v", err)
		}
		err = net.mnistPredict()
		if err != nil {
			fmt.Printf("Error while predicting: %v", err)
		}
	default:
	}
}
