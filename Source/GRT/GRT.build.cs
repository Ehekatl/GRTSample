using UnrealBuildTool;

public class GRT : ModuleRules
{
    public GRT(ReadOnlyTargetRules Target) : base(Target)
    {

        PublicIncludePaths.AddRange(new string[] { "GRT" });
        PrivateIncludePaths.AddRange(new string[] { "GRT" });

        PublicDependencyModuleNames.AddRange(new string[] { "Core" });
        PrivateDependencyModuleNames.AddRange(new string[] { });
    }
}
