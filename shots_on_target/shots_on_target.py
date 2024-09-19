import numpy as np
import cv2
from trackers import Tracker 

class ShotCounter:
    def __init__(self, model_path):
        self.tracker = Tracker(model_path)
        self.shots_on_target_team_1 = 0
        self.shots_on_target_team_2 = 0

    def is_ball_in_goal(self, ball_bbox, goal_bbox):
        """
        Check if the ball's bounding box is inside the goalpost's bounding box.
        """
        ball_x1, ball_y1, ball_x2, ball_y2 = ball_bbox
        goal_x1, goal_y1, goal_x2, goal_y2 = goal_bbox

        # Check if the ball's bounding box is fully inside the goalpost's bounding box
        return (goal_x1 <= ball_x1 <= goal_x2 and goal_y1 <= ball_y1 <= goal_y2) and \
               (goal_x1 <= ball_x2 <= goal_x2 and goal_y1 <= ball_y2 <= goal_y2)

    def annotate_shots_on_target(self, frame, shots_on_target_team_1, shots_on_target_team_2):
        """
        Annotate the shots on target for each team on the frame.
        """
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 850), (500, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, f"Team 1 Shots on Target: {shots_on_target_team_1}", (60, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Shots on Target: {shots_on_target_team_2}", (60, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def process_video(self, video_frames, tracks):
        """
        Process each frame of the video to check if the ball is in the goalpost and increment the shots on target counter.
        """
        for frame_num, frame in enumerate(video_frames):
            ball_tracks = tracks["ball"][frame_num]
            goalpost_tracks = tracks["goalpost"][frame_num]

            # For each ball and goalpost, check if the ball is in the goalpost
            for _, ball_info in ball_tracks.items():
                ball_bbox = ball_info['bbox']

                for _, goal_info in goalpost_tracks.items():
                    goal_bbox = goal_info['bbox']

                    if self.is_ball_in_goal(ball_bbox, goal_bbox):
                        # Increment shots for team depending on which side the goalpost is located
                        goal_x1 = goal_bbox[0]  # X-coordinate of the goalpost
                        if goal_x1 < frame.shape[1] // 2:
                            self.shots_on_target_team_2 += 1  # Assume goal on the left side is for Team 2
                        else:
                            self.shots_on_target_team_1 += 1  # Goal on right side is for Team 1

            # Annotate the shots on target on the frame
            frame = self.annotate_shots_on_target(frame, self.shots_on_target_team_1, self.shots_on_target_team_2)

        return video_frames
