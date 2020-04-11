#pragma once

#include <cstdint>

#ifndef PROFILING_ENABLED
#define PROFILING_ENABLED
#endif
#ifdef inline
#undef inline
#endif

namespace Profiling
{
    inline constexpr uint32_t PROFILING_CATEGORY_SYMBOL_ENGINE = 999;
    inline constexpr uint32_t PROFILING_CATEGORY_VIDEO_CAPTURE = 998;
    inline constexpr uint32_t PROFILING_CATEGORY_DMA = 997;
    inline constexpr uint32_t PROFILING_CATEGORY_IMU = 996;
    inline constexpr uint32_t PROFILING_CATEGORY_MATCHING_ALGO = 995;
    inline constexpr uint32_t PROFILING_CATEGORY_MOSSE = 994;
    inline constexpr uint32_t PROFILING_CATEGORY_UI_RENDERING = 993;

    inline constexpr uint32_t PROFILING_CATEGORY_VIN1 = 0b1110;
    inline constexpr uint32_t PROFILING_CATEGORY_VIN2 = 0b1101;
    inline constexpr uint32_t PROFILING_CATEGORY_VIN3 = 0b1100;
    inline constexpr uint32_t PROFILING_CATEGORY_VOUT1 = 0b0001;
    inline constexpr uint32_t PROFILING_CATEGORY_VOUT2 = 0b0010;
    inline constexpr uint32_t PROFILING_CATEGORY_VOUT3 = 0b0011;
    inline constexpr uint32_t PROFILING_CATEGORY_VOUT4 = 0b0100;

    void InitProfilingCategories();

	enum class Color : uint32_t
	{
		DarkGreen = 0xFF'00'55'00,
		Green = 0xFF'00'FF'00,
        DarkSeaGreen = 0xFF'00'79'6B,
        Brown = 0xFF'99'33'00,
		Blue = 0xFF'00'00'FF,
		LightBlue = 0xFF'00'BF'FF,
		Red = 0xFF'FF'00'00,
        Purple = 0xFF'99'00'99,
        Pink = 0xFF'FF'00'FF,
		Yellow = 0xFF'FF'FF'00,
        Orange = 0xFF'FF'A5'00,
		Cyan = 0xFF'00'FF'FF,
		Black = 0xFF'00'00'00,
		DarkGray = 0xFF'53'53'53,
		Gray = 0xF0'B0'B0'B0,
		LightGray = 0xFF'DC'DC'DC
	};


    void PushRange(const char* rangeName, Color color, uint32_t category = 0);
    uint64_t StartUnthreadedRange(const char* rangeName, Color color, uint32_t category = 0);
    void EndUnthreadedRange(uint64_t rangeId);
    void PushRange(const char* rangeName);
    void PopRange();
    void SetProfilingMark(const char* markName);
    void SetProfilingMark(const char* markName, Color color, uint32_t category = 0);
    void NameCurrentThreadW(uint32_t id, const wchar_t* name);

	class ProfileScope
	{
	public:
        ProfileScope(const char* rangeName, Color color, uint32_t category = 0);
        ProfileScope(const char* rangeName);
        ~ProfileScope();

        ProfileScope(const ProfileScope&) = delete;
        ProfileScope(ProfileScope&&) = delete;
        ProfileScope& operator=(ProfileScope&&) = delete;
        ProfileScope& operator=(const ProfileScope&) = delete;
	};


    class ProfileUnthreadedScope
    {
    public:
        ProfileUnthreadedScope(const char* rangeName, Color color, uint32_t category = 0);
        ~ProfileUnthreadedScope();

        ProfileUnthreadedScope(const ProfileUnthreadedScope&) = delete;
        ProfileUnthreadedScope(ProfileUnthreadedScope&&) = delete;
        ProfileUnthreadedScope& operator=(ProfileUnthreadedScope&&) = delete;
        ProfileUnthreadedScope& operator=(const ProfileUnthreadedScope&) = delete;
    private:
        uint64_t _id;
    };

}

#ifndef CONCAT_
    #define CONCAT_(x,y) x##y
    #define CONCAT(x,y) CONCAT_(x,y)
#endif


#ifdef PROFILING_ENABLED
#define PROFILE_START_CATEGORIZED_UNTHREADED_RANGE(name, color, category) Profiling::StartUnthreadedRange(name,color,category)
#define PROFILE_END_UNTHREADED_RANGE() Profiling::EndUnthreadedRange()
#define PROFILE_PUSH_RANGE(name, color) Profiling::PushRange(name,color)
#define PROFILE_PUSH_CATEGORIZED_RANGE(name, color, category) Profiling::PushRange(name,color, category)
#define PROFILE_POP_RANGE() Profiling::PopRange()
#define PROFILE_SCOPE(name, color) Profiling::ProfileScope CONCAT(__profilescope__,__COUNTER__)(name,color)
#define PROFILE_CATEGORIZED_SCOPE(name, color, category) Profiling::ProfileScope CONCAT(__profilescope__,__COUNTER__)(name, color, category)
#define PROFILE_SET_MARK(name, color) Profiling::SetProfilingMark(name,color)
#define PROFILE_SET_CATEGORIZED_MARK(name, color, category) Profiling::SetProfilingMark(name,color, category)
#define PROFILE_CATEGORIZED_UNTHREADED_SCOPE(name, color, category) Profiling::ProfileUnthreadedScope CONCAT(__profilescope__,__COUNTER__)(name, color, category)
#else
#define PROFILE_START_CATEGORIZED_UNTHREADED_RANGE(name, color, category)
#define PROFILE_END_UNTHREADED_RANGE()
#define PROFILE_PUSH_RANGE(name, color)
#define PROFILE_PUSH_CATEGORIZED_RANGE(name, color, category)
#define PROFILE_POP_RANGE() Profiling::PopRange()
#define PROFILE_SCOPE(name, color)
#define PROFILE_CATEGORIZED_SCOPE(name, color, category)
#define PROFILE_SET_MARK(name, color)
#define PROFILE_SET_CATEGORIZED_MARK(name, color, category)
#define PROFILE_CATEGORIZED_UNTHREADED_SCOPE(name, color, category)
#endif