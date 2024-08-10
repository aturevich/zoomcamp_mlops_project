provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "data_lake_bucket" {
  name     = "${var.project_id}-data-lake"
  location = var.region

  # Optional, but recommended settings:
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30  // days
    }
  }

  force_destroy = true
}

resource "google_bigquery_dataset" "dataset" {
  dataset_id = var.bq_dataset
  project    = var.project_id
  location   = var.region
}

resource "google_compute_instance" "vm_instance" {
  name         = "${var.project_id}-vm"
  machine_type = "e2-medium"
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network = "default"

    access_config {
      // Ephemeral public IP
    }
  }
}
