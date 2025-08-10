#!/usr/bin/env python3
"""
HackRx 6.0 Enhanced Compliance Test Script
Tests the system against exact HackRx specifications with detailed reporting
"""

import asyncio
import aiohttp
import json
import time
import sys
from datetime import datetime

# ALWAYS REPLACE WITH THE NEW NGROK URL
class HackRxComplianceTest:
    def __init__(self, base_url: str = "https://0abe2a16e93c.ngrok-free.app"):
        self.base_url = base_url
        self.token = "6ca800c46dd70bb4a8ef18a01692ac76721bb2b50303e31dbed18a186993ac1e"
        self.test_results = []
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result with details"""
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    async def test_hackrx_compliance(self):
        """Test complete HackRx compliance with detailed reporting"""
        print("🧪 HACKRX 6.0 COMPREHENSIVE COMPLIANCE TEST")
        print("=" * 60)
        print(f"🎯 Target URL: {self.base_url}")
        print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Test 1: Server Health Check
        print("\n1. 🏥 Testing server health...")
        health_ok = await self._test_server_health()
        self.log_test("Server Health", health_ok)
        print(f"   ✓ Server responsive: {'PASS' if health_ok else 'FAIL'}")
        
        # Test 2: API Endpoint
        print("\n2. 🎯 Testing API endpoint...")
        endpoint_correct = await self._test_endpoint()
        self.log_test("API Endpoint", endpoint_correct)
        print(f"   ✓ Endpoint /api/v1/hackrx/run: {'PASS' if endpoint_correct else 'FAIL'}")
        
        # Test 3: Authentication
        print("\n3. 🔐 Testing authentication...")
        auth_works = await self._test_authentication()
        self.log_test("Authentication", auth_works)
        print(f"   ✓ Bearer token auth: {'PASS' if auth_works else 'FAIL'}")
        
        # Test 4: Request format validation
        print("\n4. 📝 Testing request format...")
        request_works = await self._test_request_format()
        self.log_test("Request Format", request_works)
        print(f"   ✓ Request format validation: {'PASS' if request_works else 'FAIL'}")
        
        # Test 5: Response format compliance
        print("\n5. 📨 Testing response format...")
        response_format_ok = await self._test_response_format()
        self.log_test("Response Format", response_format_ok)
        print(f"   ✓ Response format compliance: {'PASS' if response_format_ok else 'FAIL'}")
        
        # Test 6: Performance test
        print("\n6. ⚡ Testing performance...")
        performance_ok = await self._test_performance()
        self.log_test("Performance", performance_ok)
        print(f"   ✓ Response time < 60s: {'PASS' if performance_ok else 'FAIL'}")
        
        # Test 7: HackRx official sample
        print("\n7. 🎪 Testing with HackRx official sample...")
        sample_works = await self._test_hackrx_sample()
        self.log_test("HackRx Sample", sample_works)
        print(f"   ✓ HackRx official sample: {'PASS' if sample_works else 'FAIL'}")
        
        # Test 8: Edge cases
        print("\n8. 🔬 Testing edge cases...")
        edge_cases_ok = await self._test_edge_cases()
        self.log_test("Edge Cases", edge_cases_ok)
        print(f"   ✓ Edge case handling: {'PASS' if edge_cases_ok else 'FAIL'}")
        
        # Generate final report
        await self._generate_final_report()
        
        # Overall compliance check
        all_tests_pass = all(result['passed'] for result in self.test_results)
        return all_tests_pass
    
    async def _test_server_health(self):
        """Test if server is running and responsive"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/v1/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return False
    
    async def _test_endpoint(self):
        """Test if endpoint exists and accepts requests"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v1/hackrx/run",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json={"documents": "test", "questions": ["test"]},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    # Should not return 404 (not found) or 405 (method not allowed)
                    return response.status not in [404, 405]
        except Exception as e:
            print(f"   ❌ Endpoint test failed: {e}")
            return False
    
    async def _test_authentication(self):
        """Test authentication requirements"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test 1: No authorization header
                async with session.post(
                    f"{self.base_url}/api/v1/hackrx/run",
                    json={"documents": "test", "questions": ["test"]},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 401:
                        print(f"   ⚠️ Expected 401 without auth, got {response.status}")
                        return False
                
                # Test 2: Invalid token
                async with session.post(
                    f"{self.base_url}/api/v1/hackrx/run",
                    headers={"Authorization": "Bearer invalid_token_12345"},
                    json={"documents": "test", "questions": ["test"]},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 401:
                        print(f"   ⚠️ Expected 401 with invalid token, got {response.status}")
                        return False
                
                # Test 3: Valid token should not return 401
                async with session.post(
                    f"{self.base_url}/api/v1/hackrx/run",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json={"documents": "https://example.com/test.pdf", "questions": ["test"]},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 401:
                        print(f"   ⚠️ Valid token returned 401")
                        return False
                
                return True
        except Exception as e:
            print(f"   ❌ Auth test failed: {e}")
            return False
    
    async def _test_request_format(self):
        """Test request format validation"""
        try:
            async with aiohttp.ClientSession() as session:
                # Valid request format
                valid_request = {
                    "documents": "https://example.com/sample.pdf",
                    "questions": ["What is this document about?", "Who is the author?"]
                }
                
                async with session.post(
                    f"{self.base_url}/api/v1/hackrx/run",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json=valid_request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    return response.status != 422  # Should not be validation error
        except Exception as e:
            print(f"   ❌ Request format test failed: {e}")
            return False
    
    async def _test_response_format(self):
        """Test response format compliance"""
        try:
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
                    "questions": ["What is the policy about?"]
                }
                
                async with session.post(
                    f"{self.base_url}/api/v1/hackrx/run",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check response structure
                        if not isinstance(data.get("answers"), list):
                            print(f"   ❌ Response missing 'answers' array")
                            return False
                        
                        # Check that answers count matches questions count
                        answers = data.get("answers", [])
                        if len(answers) != len(request_data["questions"]):
                            print(f"   ⚠️ Answer count ({len(answers)}) != question count ({len(request_data['questions'])})")
                            return False
                        
                        # Check that answers are strings
                        for i, answer in enumerate(answers):
                            if not isinstance(answer, str):
                                print(f"   ❌ Answer {i+1} is not a string: {type(answer)}")
                                return False
                        
                        return True
                    else:
                        print(f"   ❌ Response status: {response.status}")
                        return False
        except Exception as e:
            print(f"   ❌ Response format test failed: {e}")
            return False
    
    async def _test_performance(self):
        """Test response time performance"""
        try:
            request_data = {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
                "questions": ["What is covered by this policy?"]
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v1/hackrx/run",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    response_time = time.time() - start_time
                    
                    print(f"   ⏱️ Response time: {response_time:.2f} seconds")
                    
                    # HackRx typically expects responses within 60 seconds
                    if response_time > 60:
                        print(f"   ⚠️ Response time exceeded 60 seconds")
                        return False
                    
                    return response.status == 200
        except asyncio.TimeoutError:
            print(f"   ❌ Request timed out")
            return False
        except Exception as e:
            print(f"   ❌ Performance test failed: {e}")
            return False
    
    async def _test_hackrx_sample(self):
        """Test with official HackRx sample request"""
        try:
            # Official HackRx sample request
            hackrx_sample = {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What is the waiting period for pre-existing diseases?",
                    "What are the exclusions mentioned in the policy?"
                ]
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v1/hackrx/run",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json", 
                        "Authorization": f"Bearer {self.token}"
                    },
                    json=hackrx_sample,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    response_time = time.time() - start_time
                    
                    print(f"   ⏱️ Processing time: {response_time:.2f}s")
                    
                    if response.status == 200:
                        data = await response.json()
                        answers = data.get("answers", [])
                        
                        print(f"   📊 Questions sent: {len(hackrx_sample['questions'])}")
                        print(f"   📨 Answers received: {len(answers)}")
                        
                        # Show sample answers
                        for i, answer in enumerate(answers[:2]):  # Show first 2
                            print(f"   💬 Answer {i+1}: {answer[:80]}{'...' if len(answer) > 80 else ''}")
                        
                        # Check for generic/error responses
                        generic_responses = [
                            "error processing",
                            "no relevant documents",
                            "unable to process",
                            "insufficient evidence"
                        ]
                        
                        valid_answers = 0
                        for answer in answers:
                            if not any(generic in answer.lower() for generic in generic_responses):
                                if len(answer) > 20:  # Substantial answer
                                    valid_answers += 1
                        
                        print(f"   ✅ Valid answers: {valid_answers}/{len(answers)}")
                        
                        return len(answers) == len(hackrx_sample["questions"]) and valid_answers > 0
                    else:
                        print(f"   ❌ Status: {response.status}")
                        error_text = await response.text()
                        print(f"   📄 Error: {error_text[:200]}...")
                        return False
                        
        except Exception as e:
            print(f"   ❌ HackRx sample test failed: {e}")
            return False
    
    async def _test_edge_cases(self):
        """Test edge cases and error handling"""
        edge_cases_passed = 0
        total_edge_cases = 3
        
        try:
            async with aiohttp.ClientSession() as session:
                # Edge case 1: Invalid document URL
                print(f"   🧪 Testing invalid document URL...")
                try:
                    async with session.post(
                        f"{self.base_url}/api/v1/hackrx/run",
                        headers={"Authorization": f"Bearer {self.token}"},
                        json={"documents": "https://invalid-url-12345.com/nonexistent.pdf", "questions": ["test"]},
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        # Should handle gracefully (not crash)
                        if response.status in [200, 400]:  # Either success or handled error
                            edge_cases_passed += 1
                            print(f"      ✅ Invalid URL handled gracefully")
                        else:
                            print(f"      ⚠️ Unexpected status for invalid URL: {response.status}")
                except Exception as e:
                    print(f"      ❌ Invalid URL test crashed: {e}")
                
                # Edge case 2: Empty questions list
                print(f"   🧪 Testing empty questions...")
                try:
                    async with session.post(
                        f"{self.base_url}/api/v1/hackrx/run",
                        headers={"Authorization": f"Bearer {self.token}"},
                        json={"documents": "https://example.com/test.pdf", "questions": []},
                        timeout=aiohttp.ClientTimeout(total=20)
                    ) as response:
                        if response.status in [200, 400, 422]:  # Handled appropriately
                            edge_cases_passed += 1
                            print(f"      ✅ Empty questions handled")
                        else:
                            print(f"      ⚠️ Unexpected status for empty questions: {response.status}")
                except Exception as e:
                    print(f"      ❌ Empty questions test failed: {e}")
                
                # Edge case 3: Very long question
                print(f"   🧪 Testing very long question...")
                try:
                    long_question = "What is " + "really " * 100 + "covered by this policy?"
                    async with session.post(
                        f"{self.base_url}/api/v1/hackrx/run",
                        headers={"Authorization": f"Bearer {self.token}"},
                        json={"documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf", "questions": [long_question]},
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 200:
                            edge_cases_passed += 1
                            print(f"      ✅ Long question handled")
                        else:
                            print(f"      ⚠️ Long question status: {response.status}")
                except Exception as e:
                    print(f"      ❌ Long question test failed: {e}")
        
        except Exception as e:
            print(f"   ❌ Edge cases test failed: {e}")
        
        success_rate = edge_cases_passed / total_edge_cases
        print(f"   📊 Edge cases passed: {edge_cases_passed}/{total_edge_cases} ({success_rate:.1%})")
        
        return success_rate >= 0.67  # At least 2/3 should pass
    
    async def _generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📋 DETAILED TEST REPORT")
        print("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        total_tests = len(self.test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"✅ Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"⏰ Test Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📊 Test Results:")
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"   {result['test']}: {status}")
            if result['details']:
                print(f"      Details: {result['details']}")
        
        print("\n🎯 COMPLIANCE STATUS:")
        if success_rate == 100:
            print("   🏆 PERFECT COMPLIANCE - Ready for HackRx submission!")
        elif success_rate >= 85:
            print("   ✅ GOOD COMPLIANCE - Likely ready for HackRx submission")
        elif success_rate >= 70:
            print("   ⚠️  PARTIAL COMPLIANCE - Some issues need fixing")
        else:
            print("   ❌ POOR COMPLIANCE - Significant issues need resolution")

async def main():
    """Run the compliance test suite"""
    print("🚀 Starting HackRx 6.0 Compliance Test Suite...")
    
    tester = HackRxComplianceTest()
    
    try:
        compliance_passed = await tester.test_hackrx_compliance()
        
        print("\n" + "=" * 60)
        if compliance_passed:
            print("🎉 CONGRATULATIONS! Your system is HackRx compliant!")
            print("✅ Ready for official submission!")
        else:
            print("⚠️  Some compliance issues detected.")
            print("📝 Please review the test results and fix any FAIL items.")
        
        print("\n🔗 Next steps:")
        print("   1. Fix any failing tests")
        print("   2. Run this test again until all tests pass")
        print("   3. Submit to HackRx evaluation platform")
        
        return 0 if compliance_passed else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
