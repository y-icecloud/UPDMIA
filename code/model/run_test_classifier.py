from ultralytics import YOLO
import csv

def run_test_classifier(args):
    model = YOLO(args.cla_model_path)

    outputs = model(args.cla_test_dir)

    output_csv = './cla_pred.csv'

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'result'])

        for idx, result in enumerate(outputs, start=1):
            writer.writerow([idx, result.probs.top1 + 1])

    print(f"Results saved to: {output_csv}")