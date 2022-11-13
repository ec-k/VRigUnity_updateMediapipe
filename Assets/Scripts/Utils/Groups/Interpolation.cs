﻿using UnityEngine;
using System.Linq;

namespace HardCoded.VRigUnity {
	public struct RotStruct {
		public static float TestInterpolationValue => HolisticTrackingSolution.TestInterpolationValue;

		public static RotStruct identity => new(Quaternion.identity, 0);

		private float lastTime;
		private float currTime;
		private Quaternion curr;

		// Cache values
		private Transform lastTransform;
		private HumanBodyBones lastBone;

		public RotStruct(Quaternion init, float time) {
			currTime = time;
			lastTime = time;
			curr = init;

			lastTransform = null;
			lastBone = HumanBodyBones.LastBone;
		}

		public void Set(Quaternion value, float time) {
			lastTime = currTime;
			currTime = time;
			curr = value;
		}

		public Quaternion Get() {
			return curr;
		}

		private Quaternion GetUpdatedRotation(Quaternion current, Quaternion curr, float time) {
			return Quaternion.Slerp(current, curr, TestInterpolationValue);
		}

		private Transform GetTransform(Animator animator, HumanBodyBones bone) {
			if (lastTransform == null || lastBone != bone) {
				lastBone = bone;
				lastTransform = animator.GetBoneTransform(bone);
			}

			return lastTransform;
		}
			
		public void UpdateRotation(Animator animator, HumanBodyBones bone, float time) {
			Transform transform = GetTransform(animator, bone);
			if (time - 1 > currTime) {
				// If the part was lost we slowly put it back to it's original position
				transform.localRotation = Quaternion.Slerp(transform.localRotation, Quaternion.identity, 0.1f);
			} else {
				transform.rotation = GetUpdatedRotation(transform.rotation, curr, time);
			}
		}

		public Quaternion GetRawUpdateRotation(Transform transform, float time) {
			return GetUpdatedRotation(transform.rotation, curr, time);
		}

		public void UpdateLocalRotation(Animator animator, HumanBodyBones bone, float time) {
			Transform transform = GetTransform(animator, bone);
			transform.localRotation = GetUpdatedRotation(transform.localRotation, curr, time);
		}
	}

	public struct PosStruct {
		public static PosStruct identity => new(Vector3.zero, 0);
		
		private float lastTime;
		private float currTime;
		private Vector3 curr;

		// Cache values
		private Transform lastTransform;
		private HumanBodyBones lastBone;

		public PosStruct(Vector3 init, float time) {
			currTime = time;
			lastTime = time;
			curr = init;

			lastTransform = null;
			lastBone = HumanBodyBones.LastBone;
		}

		public void Set(Vector3 value, float time) {
			lastTime = currTime;
			currTime = time;
			curr = value;
		}

		public Vector3 Get() {
			return curr;
		}

		private Vector3 GetUpdatedPosition(Vector3 current, Vector3 curr, float time) {
			return Vector3.Lerp(current, curr, RotStruct.TestInterpolationValue);
		}
		
		private Transform GetTransform(Animator animator, HumanBodyBones bone) {
			if (lastTransform == null || lastBone != bone) {
				lastBone = bone;
				lastTransform = animator.GetBoneTransform(bone);
			}

			return lastTransform;
		}
		
		public void UpdatePosition(Animator animator, HumanBodyBones bone, float time) {
			Transform transform = GetTransform(animator, bone);
			if (time - 1 > currTime) {
				transform.position = Vector3.Lerp(transform.position, curr, 0.1f);
			} else {
				transform.position = GetUpdatedPosition(transform.position, curr, time);
			}
		}

		
		public Vector3 GetRawUpdatePosition(Vector3 last, float time) {
			return GetUpdatedPosition(last, curr, time);
		}
	}

	public class HandValues {
		public RotStruct Wrist = RotStruct.identity;
		public RotStruct IndexPip = RotStruct.identity;
		public RotStruct IndexDip = RotStruct.identity;
		public RotStruct IndexTip = RotStruct.identity;
		public RotStruct MiddlePip = RotStruct.identity;
		public RotStruct MiddleDip = RotStruct.identity;
		public RotStruct MiddleTip = RotStruct.identity;
		public RotStruct RingPip = RotStruct.identity;
		public RotStruct RingDip = RotStruct.identity;
		public RotStruct RingTip = RotStruct.identity;
		public RotStruct PinkyPip = RotStruct.identity;
		public RotStruct PinkyDip = RotStruct.identity;
		public RotStruct PinkyTip = RotStruct.identity;
		public RotStruct ThumbPip = RotStruct.identity;
		public RotStruct ThumbDip = RotStruct.identity;
		public RotStruct ThumbTip = RotStruct.identity;
	}

	public class PoseValues {
		public RotStruct Neck = RotStruct.identity;
		public RotStruct Chest = RotStruct.identity;
		public RotStruct Hips = RotStruct.identity;
		public PosStruct HipsPosition = PosStruct.identity;
		public RotStruct RightUpperArm = RotStruct.identity;
		public RotStruct RightLowerArm = RotStruct.identity;
		public RotStruct LeftUpperArm = RotStruct.identity;
		public RotStruct LeftLowerArm = RotStruct.identity;
		public RotStruct RightUpperLeg = RotStruct.identity;
		public RotStruct RightLowerLeg = RotStruct.identity;
		public RotStruct LeftUpperLeg = RotStruct.identity;
		public RotStruct LeftLowerLeg = RotStruct.identity;
	}

	public class FaceData {
		public struct RollingAverage {
			public float[] data;
			private int dataIndex;

			public RollingAverage(int size) {
				data = new float[size];
				dataIndex = 0;
			}

			public void Add(float value) {
				data[dataIndex] = value;
				dataIndex = (dataIndex + 1) % data.Length;
			}

			public float Average() {
				return data.Average();
			}

			public float Min() {
				return data.Min();
			}

			public float Max() {
				return data.Max();
			}
		}

		public struct RollingAverageVector2 {
			public Vector2[] data;
			private int dataIndex;

			public RollingAverageVector2(int size) {
				data = new Vector2[size];
				dataIndex = 0;
			}

			public void Add(Vector2 value) {
				data[dataIndex] = value;
				dataIndex = (dataIndex + 1) % data.Length;
			}

			public Vector2 Average() {
				return data.Aggregate((a, b) => a + b) / (float)data.Length;
			}
		}
	}
}
