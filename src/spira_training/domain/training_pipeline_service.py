class TrainingPipelineService:
    def __init__(self, training_pipeline, trigger_detector):
        self.training_pipeline = training_pipeline
        self.trigger_detector = trigger_detector

    def loop(self):
        while True:
            if self.trigger_detector.is_triggered():
                self.training_pipeline.execute()
            

    