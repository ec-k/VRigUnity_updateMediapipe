using Mediapipe;
using Mediapipe.Unity;
using System;
using UnityEngine;

namespace HardCoded.VRigUnity {
	public class HolisticDebugSolution : HolisticSolution {
		[Header("Debug")]
		[SerializeField] private HandGroup handGroup;
		[SerializeField] private int fps = 60;
		[SerializeField] private bool renderUpdate;

		private readonly Groups.HandPoints handPoints = new();
		private bool hasHandData;

		protected override void OnStartRun() {
			base.OnStartRun();
			graphRunner.OnFaceLandmarksOutput += OnFaceLandmarksOutput;
			graphRunner.OnPoseLandmarksOutput += OnPoseLandmarksOutput;
			graphRunner.OnLeftHandLandmarksOutput += OnLeftHandLandmarksOutput;
			graphRunner.OnRightHandLandmarksOutput += OnRightHandLandmarksOutput;
			graphRunner.OnPoseWorldLandmarksOutput += OnPoseWorldLandmarksOutput;
		}

		private void OnPoseLandmarksOutput(object stream, OutputEventArgs<NormalizedLandmarkList> eventArgs) {}
		private void OnFaceLandmarksOutput(object stream, OutputEventArgs<NormalizedLandmarkList> eventArgs) {}
		private void OnLeftHandLandmarksOutput(object stream, OutputEventArgs<NormalizedLandmarkList> eventArgs) {}
		private void OnPoseWorldLandmarksOutput(object stream, OutputEventArgs<LandmarkList> eventArgs) {}
		
		private void OnRightHandLandmarksOutput(object stream, OutputEventArgs<NormalizedLandmarkList> eventArgs) {
			if (eventArgs.value == null) {
				return;
			}

			int count = eventArgs.value.Landmark.Count;
			for (int i = 0; i < count; i++) {
				NormalizedLandmark mark = eventArgs.value.Landmark[i];
				handPoints.Data[i] = new(mark.X * 2, -mark.Y, -mark.Z);
			}

			hasHandData = true;
		}

		public override void Update() {
			base.Update();

			if (Application.targetFrameRate != fps) {
				Application.targetFrameRate = fps;
			}
		}

		[Range(0, 1)]
		public float angleTest = 0;
		public int test;

		public override void UpdateModel() {
			if (!renderUpdate) {	
				base.UpdateModel();
			}
		}

		public override void AnimateModel() {
			if (handGroup != null && handPoints != null) {
				handGroup.Apply(handPoints, model.ModelBones[HumanBodyBones.LeftHand].transform.position, 0.5f);
			}

			if (renderUpdate) {
				float time = TimeNow;
				RightHand.Update(time);
				LeftHand.Update(time);
				Pose.Update(time);
			}

			// Debug
			if (hasHandData) {
				HandResolver.SolveRightHand(handPoints);
			}

			base.AnimateModel();
		}
	}
}
