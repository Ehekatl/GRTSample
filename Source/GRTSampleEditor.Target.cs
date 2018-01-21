// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.Collections.Generic;

public class GRTSampleEditorTarget : TargetRules
{
	public GRTSampleEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;

		ExtraModuleNames.AddRange( new string[] { "GRTSample", "GRT" } );
	}

    //public override void SetupBinaries(
    //    TargetInfo Target,
    //    ref List<UEBuildBinaryConfiguration> OutBuildBinaryConfigurations,
    //    ref List<string> OutExtraModuleNames
    //    )
    //{
    //    OutExtraModuleNames.AddRange(new string[] { "GRTSample", "GRT" });
    //}
}
