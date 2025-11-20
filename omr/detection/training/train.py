def main();
    with open("config.json", "r") as f:
        config = json.load(f)

    print(config)
    params = {
        "lr": config["lr"],
        "batch": config["batch"],
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
    }


if __name__ == "__main__":
    main()